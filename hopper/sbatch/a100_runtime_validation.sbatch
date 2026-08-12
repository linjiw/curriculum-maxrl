#!/bin/bash
#SBATCH --job-name=maxrl-a100-validate
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=122G
#SBATCH --time=0-03:00:00
#SBATCH --output=/scratch/%u/curriculum-maxrl-runtime/logs/a100_validate_%j.out

# ENGINEERING VALIDATION — NOT EVIDENCE. Proves the full Countdown inference
# stack (vLLM + lab clean-SFT checkpoint + real train prompts) on a Hopper
# A100.80gb and measures throughput at the E2c-prereg generation shape
# (60 batches x 8 prompts x 16 rollouts, max 128 new tokens). Output sizes
# the gate-replication study; it feeds no confirmatory endpoint.
set -euo pipefail
ENVPY="/scratch/$USER/envs/maxrl/bin/python"
export HF_HOME="/scratch/$USER/.hf"
RUNTIME="/scratch/$USER/curriculum-maxrl-runtime"

"$ENVPY" - <<'EOF'
import json, time, glob
import pandas as pd
from vllm import LLM, SamplingParams

RUNTIME = "/scratch/" + __import__("os").environ["USER"] + "/curriculum-maxrl-runtime"
MODEL = RUNTIME + "/models/countdown_sft_clean_v1"

df = pd.read_parquet(RUNTIME + "/data/countdown_v2_rebuilt/train.parquet")
col = next(c for c in ["prompt","question","query","input"] if c in df.columns)
prompts = [str(p) for p in df[col].head(480).tolist()]  # 60 batches x 8
print(f"dataset rows={len(df)} prompt_col={col!r} using {len(prompts)} prompts")

llm = LLM(model=MODEL, gpu_memory_utilization=0.5, dtype="bfloat16")
sp = SamplingParams(n=16, temperature=1.0, top_p=1.0, max_tokens=128)

t0 = time.time()
tok_out = 0
for b in range(60):
    batch = prompts[b*8:(b+1)*8]
    outs = llm.generate(batch, sp, use_tqdm=False)
    tok_out += sum(len(c.token_ids) for o in outs for c in o.outputs)
    if b % 10 == 0:
        el = time.time()-t0
        print(f"batch {b:02d} | elapsed {el:6.1f}s | out-tokens {tok_out}")
el = time.time()-t0
print(json.dumps({
  "stamp": "engineering_validation_not_evidence",
  "shape": "60x8x16 max128",
  "elapsed_s": round(el,1),
  "output_tokens": tok_out,
  "tok_per_s": round(tok_out/el,1),
  "per_training_step_generation_s_estimate": round(el/60,2),
}, indent=2))
print("A100 VALIDATION PASS")
EOF
