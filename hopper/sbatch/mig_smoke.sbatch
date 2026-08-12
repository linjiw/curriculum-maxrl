#!/bin/bash
#SBATCH --job-name=maxrl-mig-smoke
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --time=0-00:30:00
#SBATCH --output=/scratch/%u/curriculum-maxrl-runtime/logs/mig_smoke_%j.out

# Cheapest end-to-end validation of the Hopper GPU stack. Submit FIRST.
# Pass criteria: CUDA device visible, model loads, 5 greedy generations emitted.
# Uses the scratch-prefixed conda env directly (home quota is nearly full).
set -euo pipefail

ENVPY="/scratch/$USER/envs/maxrl/bin/python"
export HF_HOME="/scratch/$USER/.hf"
mkdir -p "$HF_HOME"

"$ENVPY" - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not visible"
print("device:", torch.cuda.get_device_name(0),
      "| mem:", round(torch.cuda.get_device_properties(0).total_memory/2**30, 1), "GiB")

from transformers import AutoModelForCausalLM, AutoTokenizer
name = "HuggingFaceTB/SmolLM2-360M"  # public base; real runs use the lab SFT checkpoint
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).cuda()
x = tok("Countdown: reach 24 using 3, 8, 8:", return_tensors="pt").to("cuda")
for i in range(5):
    y = model.generate(**x, max_new_tokens=32, do_sample=False)
    print(i, tok.decode(y[0][x['input_ids'].shape[1]:], skip_special_tokens=True)[:60])
print("SMOKE PASS")
EOF
