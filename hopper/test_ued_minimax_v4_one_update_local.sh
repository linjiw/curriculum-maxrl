#!/usr/bin/env bash
# Real pinned-CPU Rung-2 execution in a fresh v4-applied clone.
set -euo pipefail
umask 077
readonly HERE="$(cd "$(dirname "$0")" && pwd)"
readonly ROOT="$(cd "$HERE/.." && pwd)"
readonly SOURCE="${MINIMAX_SOURCE_DIR:-/tmp/root-minimax-260814}"
readonly PY="${UED_CPU_PYTHON:-/data/robotixx/ued_bench/envs/minimax-jax0431-cpu/bin/python}"
[[ -x "$PY" && ( -d "$SOURCE/.git" || -f "$SOURCE/.git" ) ]]
readonly TMP="$(mktemp -d /tmp/ued-v4-r2-local.XXXXXX)"
cleanup() { [[ "$TMP" == /tmp/ued-v4-r2-local.* && -d "$TMP" ]] && rm -rf -- "$TMP"; }
trap cleanup EXIT
git clone --quiet --no-hardlinks "$SOURCE" "$TMP/minimax"
git -C "$TMP/minimax" checkout --quiet --detach d053054c5290a04c1c4cd8b55704d999cad73e30
"$PY" -I -B "$ROOT/ued_benchmark/scripts/apply_minimax_overlay_v4.py" --target "$TMP/minimax" --check > "$TMP/check.json"
"$PY" -I -B "$ROOT/ued_benchmark/scripts/apply_minimax_overlay_v4.py" --target "$TMP/minimax" --apply > "$TMP/apply.json"
"$PY" -I -B "$ROOT/ued_benchmark/scripts/apply_minimax_overlay_v4.py" --target "$TMP/minimax" --check > "$TMP/postcheck.json"
grep -Fq '"status": "already_applied"' "$TMP/postcheck.json"
[[ "$(sha256sum "$TMP/minimax/.frontierrl_overlay.json" | awk '{print $1}')" == 9b411f61ebc56bb93fc22cad6b19299c38eab2b696fa17f7783c7729e1db02ae ]]
git -C "$TMP/minimax" diff --check
mkdir "$TMP/output"
UED_PROVENANCE="$TMP/provenance.json" UED_APPLIED="$TMP/minimax/.frontierrl_overlay.json" \
python3 -I -B - <<'PY'
import hashlib, json, os
from pathlib import Path
fake = "a"*64
fields = (
 "bundle_manifest_sha256", "upstream_git_bundle_sha256", "overlay_manifest_sha256",
 "sbatch_sha256", "environment_lock_sha256", "environment_freeze_sha256",
 "environment_manifest_sha256", "environment_setup_script_sha256",
 "conda_explicit_sha256", "environment_json_sha256", "import_smoke_manifest_sha256",
 "import_smoke_sbatch_sha256")
hashes = {field: fake for field in fields}
hashes.update({
 "applied_overlay_manifest_sha256": hashlib.sha256(Path(os.environ["UED_APPLIED"]).read_bytes()).hexdigest(),
 "config_sha256": "0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2",
 "overlay_contract_sha256": "3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b",
 "import_smoke_bundle_manifest_sha256": fake,
 "upstream_commit": "d053054c5290a04c1c4cd8b55704d999cad73e30",
 "upstream_tree_git_sha1": "b0cace1fc54984e21a842f12d15d0b899e33d270",
})
record = {
 "provenance_schema": 1,
 "purpose": "bounded Frontier grouped one-update engineering validation",
 "paper_evidence": False, "endpoint_class": "bounded_engineering_one_update",
 "max_student_updates": 1, "git": "git version 2.45.2", "job_id": "local-test",
 "xpid": "eng1-ca-ovv4ch3d5f3827_N8ne8a1.0b1.0th0.0eastrict-rt-4p-b8-rp1-mf0.5-seed1",
 "resources": {"partition": "gpuq", "qos": "gpu", "gres": "gpu:1g.10gb:1",
  "cpus_per_task": 2, "memory": "15G", "walltime": "00:30:00"},
 "hashes": hashes,
}
Path(os.environ["UED_PROVENANCE"]).write_text(json.dumps(record, indent=2, sort_keys=True)+"\n", encoding="utf-8")
PY
readonly DRIVER="$ROOT/ued_benchmark/scripts/run_grouped_one_update_v4.py"
readonly LAUNCHER='import runpy,sys; source=sys.argv.pop(1); root=sys.argv.pop(1); script=sys.argv.pop(1); sys.path[:0]=[source,root]; sys.argv[0]=script; runpy.run_path(script,run_name="__main__")'
env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE -u LD_PRELOAD \
  JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
  "$PY" -I -B -c "$LAUNCHER" "$TMP/minimax/src" "$ROOT" "$DRIVER" \
  --config "$ROOT/ued_benchmark/configs/maze_frontier_exact_grouped_n8_tie_aware_v4.json" \
  --contract "$ROOT/ued_benchmark/OVERLAY_CONTRACT_V4.json" \
  --provenance "$TMP/provenance.json" --patched-source-dir "$TMP/minimax" \
  --output-dir "$TMP/output" --local-test-mode > "$TMP/stdout" 2> "$TMP/stderr"
grep -Fq 'GROUPED_ONE_UPDATE_PASS updates=1 trials=64' "$TMP/stdout"
UED_RESULT="$TMP/output/run-result.json" UED_CHECKPOINT="$TMP/output/checkpoint.pkl" \
python3 -I -B - <<'PY'
import hashlib, json, math, os
from pathlib import Path
result = json.load(open(os.environ["UED_RESULT"], encoding="utf-8"))
assert result["status"] == "passed" and result["paper_evidence"] is False
assert result["endpoint_class"] == "bounded_engineering_one_update"
assert result["max_student_updates"] == result["actual_student_updates"] == 1
assert result["engineering_schedule"]["outer_cycles"] == 2
assert result["engineering_schedule"]["actual_ppo_updates"] == 1
warmup, replay = result["cycles"]
for key in ("replay_group_draw_count", "replay_distinct_group_count",
            "replay_duplicate_group_count", "last_replay_group_count",
            "last_replay_distinct_group_count", "last_replay_duplicate_group_count"):
    assert warmup["state"][key] == 0
final = result["final_state"]
assert final["n_updates"] == final["n_grad_updates"] == 1
assert final["tie_aware_score_ranks"] is True
assert final["nonfinite_filled_score_count"] == final["nonfinite_score_rejection_count"] == 0
assert final["replay_group_draw_count"] == final["last_replay_group_count"] == 4
assert final["replay_group_draw_count"] == final["replay_distinct_group_count"] + final["replay_duplicate_group_count"]
assert final["last_replay_group_count"] == final["last_replay_distinct_group_count"] + final["last_replay_duplicate_group_count"]
for key in ("plr/score_distribution_effective_support", "plr/replay_distribution_effective_support"):
    assert math.isfinite(float(replay["selected_stats"][key]))
checkpoint = Path(os.environ["UED_CHECKPOINT"])
assert result["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert result["checkpoint"]["counter_and_buffer_continuity"] is True
assert result["checkpoint"]["train_state_exact_leaf_continuity"] is True
assert result["checkpoint"]["post_resume_update_executed"] is False
PY
echo "UED_MINIMAX_V4_R2_LOCAL_PASS"
