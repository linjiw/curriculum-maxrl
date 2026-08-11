"""Outcome-blind, read-only readiness audit for the frozen E2c driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from curriculum_maxrl.countdown.analyze_e2c import validate_seed
from curriculum_maxrl.countdown.e2c_protocol import (
    FROZEN_ASSET_SHA256,
    FROZEN_BASE_MODEL_REVISION,
    FROZEN_GPU_MEMORY_CEILING_MIB,
    FROZEN_MAXRL_COMMIT,
    FROZEN_OPTIMIZER_ROWS,
    FROZEN_SEEDS,
    FROZEN_STEPS,
)
from curriculum_maxrl.countdown.verify_e2c_code_manifest import (
    verify_code_manifest,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_RAY_LOG_PREFIX = re.compile(
    r"^\((?:TaskRunner|WorkerDict) pid=\d+\)\s*")


def normalized_training_log(path: Path) -> str:
    """Remove terminal/Ray decoration while retaining the logged config."""
    lines = []
    raw = path.read_text(encoding="utf-8", errors="replace")
    for line in raw.replace("\r", "\n").splitlines():
        lines.append(_RAY_LOG_PREFIX.sub("", _ANSI_ESCAPE.sub("", line)))
    return " ".join(" ".join(lines).split())


def _log_section(text: str, start: str, end: str) -> str:
    start_index = text.find(start)
    end_index = text.find(end, start_index + len(start))
    if start_index < 0 or end_index <= start_index:
        return ""
    return text[start_index:end_index]


def parse_logged_training_metrics(text: str) -> dict[int, dict[str, float]]:
    """Recover the scalar metrics printed once for every optimizer step."""
    rows = {}
    for match in re.finditer(
            r"step:(\d+) - (.*?) - training/global_step:\1\.000", text):
        rows[int(match.group(1))] = {
            key: float(value)
            for key, value in re.findall(
                r"([A-Za-z_][A-Za-z0-9_/]*):"
                r"(-?[0-9]+(?:\.[0-9]+)?)",
                match.group(2))
        }
    return rows


def validate_comparator_configuration(
    text: str,
    runtime_root: Path,
    research_root: Path,
    run_id: str,
    arm: str,
    seed: int,
    expected_live_replay: bool = False,
) -> dict:
    """Verify that a reusable B1/B2 log encodes the frozen E2 protocol."""
    model_path = runtime_root / "models/countdown_sft_clean_v1"
    train_path = runtime_root / "data/countdown_v2_rebuilt/train.parquet"
    test_path = runtime_root / "data/countdown_v2_rebuilt/test.parquet"
    reward_path = research_root / "curriculum_maxrl/countdown/countdown_reward.py"
    output_dir = runtime_root / "checkpoints" / run_id
    actor = _log_section(text, "'actor_rollout_ref':", "'algorithm':")
    algorithm = _log_section(text, "'algorithm':", "'critic':")
    hindsight = _log_section(algorithm, "'hindsight':", "'kl_ctrl':")
    live_replay = _log_section(
        algorithm, "'live_replay':", "'norm_adv_by_std_in_grpo':")
    custom_reward = _log_section(
        text, "'custom_reward_function':", "'data':")
    data = _log_section(text, "'data':", "'diagnostics':")
    trainer_index = text.find("'trainer':")
    trainer = text[trainer_index:] if trainer_index >= 0 else ""
    hindsight_value = "True" if arm == "b2" else "False"
    live_replay_value = "True" if expected_live_replay else "False"
    logged_steps = [
        int(value) for value in re.findall(
            r"training/global_step:(\d+)\.000", text)]
    checks = {
        "actor_section_logged": bool(actor),
        "algorithm_section_logged": bool(algorithm),
        "data_section_logged": bool(data),
        "trainer_section_logged": bool(trainer),
        "source_model": f"'path': '{model_path}'" in actor,
        "actor_learning_rate_1e_5": "'optim': {'lr': 1e-05" in actor,
        "actor_no_lr_warmup_steps": "'lr_warmup_steps': -1" in actor,
        "actor_no_lr_warmup_ratio": (
            "'lr_warmup_steps_ratio': 0.0" in actor),
        "actor_min_lr_ratio_0": "'min_lr_ratio': 0.0" in actor,
        "actor_lr_cycles_0_5": "'num_cycles': 0.5" in actor,
        "actor_lr_schedule_constant": "'warmup_style': 'constant'" in actor,
        "actor_weight_decay_0_01": "'weight_decay': 0.01" in actor,
        "actor_one_ppo_epoch": "'ppo_epochs': 1" in actor,
        "actor_minibatch_8": "'ppo_mini_batch_size': 8" in actor,
        "actor_microbatch_4": (
            "'ppo_micro_batch_size_per_gpu': 4" in actor),
        "actor_no_kl_loss": "'use_kl_loss': False" in actor,
        "checkpoint_contents": (
            "'save_contents': ['model', 'optimizer', 'extra', 'hf_model']"
            in actor),
        "rollouts_16": "'n': 16" in actor,
        "rollout_backend_hf": "'name': 'hf'" in actor,
        "rollout_microbatch_128": "'micro_batch_size': 128" in actor,
        "logprob_microbatch_8": (
            "'log_prob_micro_batch_size_per_gpu': 8" in actor),
        "response_length_128": "'response_length': 128" in actor,
        "rollout_temperature_1": "'temperature': 1.0" in actor,
        "rollout_top_p_1": "'top_p': 1.0" in actor,
        "rollout_top_k_0": "'top_k': 0" in actor,
        "actor_seed": f"'seed': {seed}" in actor,
        "maxrl_estimator": "'adv_estimator': 'maxrl'" in algorithm,
        "maxrl_pass_k_16": "'pass_k': 16" in algorithm,
        "maxrl_truncate_order_16": "'truncate_order': 16" in algorithm,
        "no_kl_reward": "'use_kl_in_reward': False" in algorithm,
        "hindsight_arm_assignment": (
            bool(hindsight) and f"'enable': {hindsight_value}" in hindsight),
        "hindsight_scale_1": "'scale': 1.0" in hindsight,
        "hindsight_max_groups_8": (
            "'max_groups_per_step': 8" in hindsight),
        "hindsight_one_target_false": (
            "'one_target_per_group': False" in hindsight),
        "hindsight_utility_gate_false": (
            "'utility_gate': False" in hindsight),
        "live_replay_assignment": (
            bool(live_replay) and
            f"'enable': {live_replay_value}" in live_replay),
        "train_data": f"'train_files': '{train_path}'" in data,
        "heldout_data_configured_but_not_run": (
            f"'val_files': '{test_path}'" in data),
        "train_batch_8": "'train_batch_size': 8" in data,
        "data_seed": f"'seed': {seed}" in data,
        "dataloader_workers_0": "'dataloader_num_workers': 0" in data,
        "filter_overlong_false": "'filter_overlong_prompts': False" in data,
        "max_prompt_length_256": "'max_prompt_length': 256" in data,
        "max_response_length_128": "'max_response_length': 128" in data,
        "reward_function_name": "'name': 'compute_score'" in custom_reward,
        "reward_function_path": f"'path': '{reward_path}'" in custom_reward,
        "reward_manager_dapo": "'reward_manager': 'dapo'" in text,
        "experiment_name": f"'experiment_name': '{run_id}'" in trainer,
        "output_directory": f"'default_local_dir': '{output_dir}'" in trainer,
        "save_step_60": "'save_freq': 60" in trainer,
        "test_frequency_disabled": "'test_freq': -1" in trainer,
        "training_steps_60": "'total_training_steps': 60" in trainer,
        "validation_before_training_false": (
            "'val_before_train': False" in trainer),
        "validation_on_last_step_false": (
            "'val_on_last_step': False" in trainer),
        "one_gpu_one_node": (
            "'n_gpus_per_node': 1" in trainer and "'nnodes': 1" in trainer),
        "fresh_start": "Training from scratch" in text,
        "final_step_logged": (
            f"training/global_step:{FROZEN_STEPS}.000" in text),
        "exact_optimizer_step_sequence": (
            logged_steps == list(range(1, FROZEN_STEPS + 1))),
        "runtime_scheduler_60_steps_no_warmup": (
            "Total steps: 60, num_warmup_steps: 0" in text),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
    }


def validate_e2c_configuration(
    text: str,
    runtime_root: Path,
    research_root: Path,
    run_id: str,
    seed: int,
    reservoir_sha256: str,
) -> dict:
    """Validate the common training protocol plus E2c-only replay settings."""
    report = validate_comparator_configuration(
        text, runtime_root, research_root, run_id, "b1", seed,
        expected_live_replay=True)
    algorithm = _log_section(text, "'algorithm':", "'critic':")
    live_replay = _log_section(
        algorithm, "'live_replay':", "'norm_adv_by_std_in_grpo':")
    reservoir_config = _log_section(
        algorithm, "'replay_reservoir':", "'truncate_order':")
    checkpoint_root = runtime_root / "checkpoints"
    schedule = (checkpoint_root / f"e2_clean_b2_s{seed}_260809" /
                "dose_accounting.jsonl")
    reservoir = (checkpoint_root / "e2c_reservoir_collect_260810" /
                 "replay_reservoir.pt")
    audit = checkpoint_root / run_id / "replay_accounting.jsonl"
    report["checks"].update({
        "e2c_schedule_path": f"'schedule_path': '{schedule}'" in live_replay,
        "e2c_replay_seed": f"'seed': {seed}" in live_replay,
        "e2c_mismatch_limit": (
            "'max_cumulative_token_mismatch_fraction': 0.05" in live_replay),
        "e2c_strict": "'strict': True" in live_replay,
        "e2c_live_buffer_disabled": (
            "'buffer_capacity_groups': 0" in live_replay),
        "e2c_buffer_age_inactive_default": (
            "'max_buffer_age_steps': 8" in live_replay),
        "e2c_reservoir_path": (
            f"'reservoir_path': '{reservoir}'" in live_replay),
        "e2c_reservoir_sha256": (
            f"'reservoir_sha256': '{reservoir_sha256}'" in live_replay),
        "e2c_audit_path": f"'audit_path': '{audit}'" in live_replay,
        "e2c_collector_disabled": (
            bool(reservoir_config) and "'enable': False" in reservoir_config),
    })
    report["failed_checks"] = sorted(
        name for name, passed in report["checks"].items() if not passed)
    report["status"] = "pass" if not report["failed_checks"] else "fail"
    return report


def validate_reservoir_collection_configuration(
    text: str,
    runtime_root: Path,
    research_root: Path,
    run_id: str,
) -> dict:
    """Validate the frozen-SFT, no-update reservoir generation protocol."""
    model_path = runtime_root / "models/countdown_sft_clean_v1"
    train_path = runtime_root / "data/countdown_v2_rebuilt/train.parquet"
    test_path = runtime_root / "data/countdown_v2_rebuilt/test.parquet"
    reward_path = research_root / "curriculum_maxrl/countdown/countdown_reward.py"
    output_dir = runtime_root / "checkpoints" / run_id
    artifact = output_dir / "replay_reservoir.pt"
    actor = _log_section(text, "'actor_rollout_ref':", "'algorithm':")
    algorithm = _log_section(text, "'algorithm':", "'critic':")
    hindsight = _log_section(algorithm, "'hindsight':", "'kl_ctrl':")
    live_replay = _log_section(
        algorithm, "'live_replay':", "'norm_adv_by_std_in_grpo':")
    reservoir_config = _log_section(
        algorithm, "'replay_reservoir':", "'truncate_order':")
    custom_reward = _log_section(
        text, "'custom_reward_function':", "'data':")
    data = _log_section(text, "'data':", "'diagnostics':")
    trainer_index = text.find("'trainer':")
    trainer = text[trainer_index:] if trainer_index >= 0 else ""
    checks = {
        "source_model": f"'path': '{model_path}'" in actor,
        "learning_rate_zero": "'optim': {'lr': 0" in actor,
        "one_ppo_epoch": "'ppo_epochs': 1" in actor,
        "minibatch_8": "'ppo_mini_batch_size': 8" in actor,
        "microbatch_4": "'ppo_micro_batch_size_per_gpu': 4" in actor,
        "rollouts_16": "'n': 16" in actor,
        "rollout_backend_hf": "'name': 'hf'" in actor,
        "response_length_128": "'response_length': 128" in actor,
        "temperature_1": "'temperature': 1.0" in actor,
        "top_p_1": "'top_p': 1.0" in actor,
        "top_k_0": "'top_k': 0" in actor,
        "actor_seed_424242": "'seed': 424242" in actor,
        "maxrl_estimator": "'adv_estimator': 'maxrl'" in algorithm,
        "hindsight_disabled": "'enable': False" in hindsight,
        "live_replay_disabled": "'enable': False" in live_replay,
        "collector_enabled": "'enable': True" in reservoir_config,
        "collector_output": f"'output_path': '{artifact}'" in reservoir_config,
        "collector_seed": "'seed': 424242" in reservoir_config,
        "collector_max_groups": "'max_groups': 256" in reservoir_config,
        "collector_group_size": (
            "'expected_group_size': 16" in reservoir_config),
        "train_data": f"'train_files': '{train_path}'" in data,
        "heldout_data_configured_but_not_run": (
            f"'val_files': '{test_path}'" in data),
        "train_batch_8": "'train_batch_size': 8" in data,
        "data_seed_424242": "'seed': 424242" in data,
        "reward_function_name": "'name': 'compute_score'" in custom_reward,
        "reward_function_path": f"'path': '{reward_path}'" in custom_reward,
        "experiment_name": f"'experiment_name': '{run_id}'" in trainer,
        "output_directory": f"'default_local_dir': '{output_dir}'" in trainer,
        "checkpoint_saving_disabled": "'save_freq': -1" in trainer,
        "test_frequency_disabled": "'test_freq': -1" in trainer,
        "training_steps_60": "'total_training_steps': 60" in trainer,
        "validation_before_training_false": (
            "'val_before_train': False" in trainer),
        "validation_on_last_step_false": (
            "'val_on_last_step': False" in trainer),
        "fresh_start": "Training from scratch" in text,
        "final_step_logged": "training/global_step:60.000" in text,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
    }


def inspect_comparator(
    runtime_root: Path,
    research_root: Path,
    active_log_dir: Path,
    arm: str,
    seed: int,
) -> tuple[dict, list[str]]:
    run_id = f"e2_clean_{arm}_s{seed}_260809"
    output_dir = runtime_root / "checkpoints" / run_id
    marker = output_dir / ".complete"
    checkpoint = (output_dir / f"global_step_{FROZEN_STEPS}" / "actor" /
                  "huggingface" / "config.json")
    audit = output_dir / "dose_accounting.jsonl"
    active_log = active_log_dir / f"{run_id}.log"
    historical_log = (research_root / "autoresearch" /
                      "iterate-260809-1533" / "e2_logs" / f"{run_id}.log")
    log_path = active_log if active_log.is_file() else historical_log
    present = marker.exists() or output_dir.exists() or active_log.exists()
    issues: list[str] = []

    report = {
        "arm": arm,
        "seed": seed,
        "run_id": run_id,
        "status": "missing",
        "marker": str(marker),
        "checkpoint": str(checkpoint),
        "training_log": str(log_path),
    }
    if not marker.is_file():
        if present:
            report["status"] = "partial"
            issues.append(f"{run_id}: artifacts exist without .complete")
        return report, issues

    report["status"] = "complete"
    checkpoint_model = checkpoint.with_name("model.safetensors")
    if not checkpoint.is_file():
        issues.append(f"{run_id}: final Hugging Face checkpoint is missing")
    if not checkpoint_model.is_file():
        issues.append(f"{run_id}: final model.safetensors is missing")
    normalized_log = ""
    if not log_path.is_file():
        issues.append(f"{run_id}: final training step is absent from log")
    else:
        normalized_log = normalized_training_log(log_path)
        configuration = validate_comparator_configuration(
            normalized_log, runtime_root, research_root, run_id, arm, seed)
        report["configuration_validation"] = configuration
        report["training_log_sha256"] = sha256(log_path)
        if configuration["status"] != "pass":
            issues.append(
                f"{run_id}: frozen config checks failed: "
                f"{configuration['failed_checks']}")
    if checkpoint.is_file() and checkpoint_model.is_file():
        report["checkpoint_fingerprint"] = {
            "config.json": {
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
            },
            "model.safetensors": {
                "bytes": checkpoint_model.stat().st_size,
                "sha256": sha256(checkpoint_model),
            },
        }
    if arm == "b2":
        if not audit.is_file():
            issues.append(f"{run_id}: B2 dose schedule is missing")
        else:
            try:
                rows = read_jsonl(audit)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                rows = []
                issues.append(f"{run_id}: B2 schedule is unreadable: {error}")
            steps = [int(row.get("global_step", -1)) for row in rows]
            if steps != list(range(1, FROZEN_STEPS + 1)):
                issues.append(f"{run_id}: B2 schedule is not ordered steps 1..60")
            if any(int(row.get("hindsight/optimizer_rows_total", -1)) !=
                   FROZEN_OPTIMIZER_ROWS for row in rows):
                issues.append(f"{run_id}: B2 schedule has an optimizer-row drift")
            if any(len(row.get("accepted_groups", [])) !=
                   len(row.get("accepted_group_token_counts", []))
                   for row in rows):
                issues.append(f"{run_id}: B2 accepted-slot metadata is incomplete")
            scheduled_groups = 0
            for row in rows:
                groups = row.get("accepted_groups", [])
                counts = row.get("accepted_group_token_counts", [])
                indices = [group.get("dataset_index") for group in groups]
                group_counts = [group.get("response_tokens") for group in groups]
                if len(groups) > 8:
                    issues.append(
                        f"{run_id}: B2 schedule requests more than 8 groups")
                    break
                if any(value is None for value in indices + group_counts):
                    issues.append(
                        f"{run_id}: B2 schedule lacks slot/token metadata")
                    break
                if len(indices) != len(set(indices)):
                    issues.append(
                        f"{run_id}: B2 schedule repeats a slot within one step")
                    break
                if [int(value) for value in group_counts] != [
                        int(value) for value in counts]:
                    issues.append(
                        f"{run_id}: B2 token-count summaries disagree")
                    break
                if sum(int(value) for value in counts) != int(
                        row.get("hindsight/aux_group_response_tokens", -1)):
                    issues.append(
                        f"{run_id}: B2 auxiliary-token total is inconsistent")
                    break
                for group in groups:
                    rollout_counts = group.get("rollout_token_counts", [])
                    if (int(group.get("rollouts", -1)) != 16 or
                            len(rollout_counts) != 16 or
                            sum(int(value) for value in rollout_counts) !=
                            int(group["response_tokens"])):
                        issues.append(
                            f"{run_id}: B2 group token payload is inconsistent")
                        break
                scheduled_groups += len(groups)
            logged_metrics = parse_logged_training_metrics(normalized_log)
            schedule_metric_fields = (
                "hindsight/dead_groups",
                "hindsight/relabeled_groups",
                "hindsight/relabeled_rollouts",
                "hindsight/skipped_rewrite",
                "hindsight/aux_group_response_tokens",
                "hindsight/pre_optimizer_response_tokens_total",
                "hindsight/optimizer_rows_total",
                "hindsight/optimizer_response_tokens_total",
                "hindsight/pre_success_rollouts",
                "hindsight/post_success_rollouts",
            )
            if set(logged_metrics) != set(range(1, FROZEN_STEPS + 1)):
                issues.append(f"{run_id}: B2 step metrics are incomplete")
            else:
                for row in rows:
                    step = int(row["global_step"])
                    for field in schedule_metric_fields:
                        if (field not in logged_metrics[step] or
                                int(logged_metrics[step][field]) !=
                                int(row[field])):
                            issues.append(
                                f"{run_id}: schedule/log mismatch at step "
                                f"{step} field {field}")
                            break
                    if issues and issues[-1].startswith(
                            f"{run_id}: schedule/log mismatch"):
                        break
            report["dose_schedule"] = str(audit)
            report["dose_schedule_rows"] = len(rows)
            report["dose_schedule_sha256"] = sha256(audit)
            report["scheduled_groups"] = scheduled_groups
    return report, issues


def inspect_reservoir(
    runtime_root: Path,
    research_root: Path,
    active_log_dir: Path,
) -> tuple[dict, list[str]]:
    run_id = "e2c_reservoir_collect_260810"
    output_dir = runtime_root / "checkpoints" / run_id
    marker = output_dir / ".complete"
    artifact = output_dir / "replay_reservoir.pt"
    manifest = Path(f"{artifact}.manifest.json")
    log_path = active_log_dir / f"{run_id}.log"
    present = output_dir.exists() or log_path.exists()
    report = {"run_id": run_id, "status": "missing"}
    issues: list[str] = []
    if not marker.is_file():
        if present:
            report["status"] = "partial"
            issues.append(f"{run_id}: artifacts exist without .complete")
        return report, issues
    report["status"] = "complete"
    for path in (artifact, manifest, log_path):
        if not path.is_file():
            issues.append(f"{run_id}: missing {path}")
    if artifact.is_file() and manifest.is_file():
        try:
            with manifest.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            actual = sha256(artifact)
            if metadata.get("sha256") != actual:
                issues.append(f"{run_id}: reservoir manifest checksum mismatch")
            report["artifact"] = str(artifact)
            report["sha256"] = actual
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            issues.append(f"{run_id}: reservoir manifest is unreadable: {error}")
    if log_path.is_file():
        normalized_log = normalized_training_log(log_path)
        configuration = validate_reservoir_collection_configuration(
            normalized_log, runtime_root, research_root, run_id)
        report["configuration_validation"] = configuration
        report["training_log_sha256"] = sha256(log_path)
        if configuration["status"] != "pass":
            issues.append(
                f"{run_id}: frozen config checks failed: "
                f"{configuration['failed_checks']}")
    return report, issues


def inspect_e2c_run(
    runtime_root: Path,
    research_root: Path,
    active_log_dir: Path,
    seed: int,
) -> tuple[dict, list[str]]:
    run_id = f"e2c_reservoir_replay_s{seed}_260810"
    output_dir = runtime_root / "checkpoints" / run_id
    marker = output_dir / ".complete"
    log_path = active_log_dir / f"{run_id}.log"
    checkpoint = (output_dir / f"global_step_{FROZEN_STEPS}" / "actor" /
                  "huggingface" / "config.json")
    audit = output_dir / "replay_accounting.jsonl"
    present = output_dir.exists() or log_path.exists()
    report = {"seed": seed, "run_id": run_id, "status": "missing"}
    issues: list[str] = []
    if not marker.is_file():
        if present:
            report["status"] = "partial"
            issues.append(f"{run_id}: artifacts exist without .complete")
        return report, issues
    report["status"] = "complete"
    checkpoint_model = checkpoint.with_name("model.safetensors")
    if not checkpoint.is_file():
        issues.append(f"{run_id}: final Hugging Face checkpoint is missing")
    if not checkpoint_model.is_file():
        issues.append(f"{run_id}: final model.safetensors is missing")
    manifest = (runtime_root / "checkpoints/e2c_reservoir_collect_260810" /
                "replay_reservoir.pt.manifest.json")
    reservoir_sha256 = ""
    if manifest.is_file():
        try:
            with manifest.open(encoding="utf-8") as handle:
                reservoir_sha256 = str(json.load(handle).get("sha256", ""))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            issues.append(f"{run_id}: reservoir manifest is unreadable: {error}")
    if not log_path.is_file():
        issues.append(f"{run_id}: final training step is absent from log")
    else:
        normalized_log = normalized_training_log(log_path)
        configuration = validate_e2c_configuration(
            normalized_log, runtime_root, research_root, run_id, seed,
            reservoir_sha256)
        report["configuration_validation"] = configuration
        report["training_log_sha256"] = sha256(log_path)
        if configuration["status"] != "pass":
            issues.append(
                f"{run_id}: frozen config checks failed: "
                f"{configuration['failed_checks']}")
    if checkpoint.is_file() and checkpoint_model.is_file():
        report["checkpoint_fingerprint"] = {
            "config.json": {
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
            },
            "model.safetensors": {
                "bytes": checkpoint_model.stat().st_size,
                "sha256": sha256(checkpoint_model),
            },
        }
    if not audit.is_file():
        issues.append(f"{run_id}: replay delivery audit is missing")
    else:
        schedule = (runtime_root / "checkpoints" /
                    f"e2_clean_b2_s{seed}_260809" /
                    "dose_accounting.jsonl")
        try:
            report["delivery_validation"] = validate_seed(
                seed, audit, schedule, FROZEN_STEPS, 0.05,
                FROZEN_OPTIMIZER_ROWS)
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"{run_id}: replay delivery invalid: {error}")
    report["delivery_audit"] = str(audit)
    return report, issues


def audit_readiness(
    runtime_root: Path,
    research_root: Path,
    iteration_dir: Path,
) -> dict:
    runtime_root = runtime_root.resolve()
    research_root = research_root.resolve()
    iteration_dir = iteration_dir.resolve()
    active_log_dir = iteration_dir / "e2c_logs"
    result_dir = iteration_dir / "e2c_results"
    issues: list[str] = []

    code_manifest_path = iteration_dir / "E2C_CODE_MANIFEST.json"
    try:
        code_manifest = verify_code_manifest(
            code_manifest_path, research_root, runtime_root)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        code_manifest = {
            "status": "fail",
            "manifest": str(code_manifest_path),
            "error": str(error),
        }
        issues.append(f"E2c code manifest failed: {error}")

    train = runtime_root / "data/countdown_v2_rebuilt/train.parquet"
    test = runtime_root / "data/countdown_v2_rebuilt/test.parquet"
    model = runtime_root / "models/countdown_sft_clean_v1"
    assets = {"train.parquet": train, "test.parquet": test}
    assets.update({name: model / name for name in (
        "config.json", "model.safetensors", "tokenizer.json",
        "training_metrics.json")})
    assets["countdown_reward.py"] = (
        research_root / "curriculum_maxrl/countdown/countdown_reward.py")
    assets["eval_countdown.py"] = (
        research_root / "curriculum_maxrl/countdown/eval_countdown.py")
    asset_report = {}
    for name, path in assets.items():
        if not path.is_file():
            issues.append(f"missing frozen asset: {path}")
            asset_report[name] = {"path": str(path), "status": "missing"}
            continue
        digest = sha256(path)
        status = "pass" if digest == FROZEN_ASSET_SHA256[name] else "mismatch"
        if status != "pass":
            issues.append(f"frozen asset checksum mismatch: {path}")
        asset_report[name] = {
            "path": str(path), "sha256": digest, "status": status}

    maxrl_root = runtime_root / "maxrl"
    try:
        actual_commit = _command("git", "-C", str(maxrl_root), "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        actual_commit = "unavailable"
    if actual_commit != FROZEN_MAXRL_COMMIT:
        issues.append(
            f"runtime MaxRL commit {actual_commit} != {FROZEN_MAXRL_COMMIT}")
    vendored = research_root / "verl_integration/vendored/hindsight.py"
    runtime_hindsight = maxrl_root / "verl/utils/hindsight.py"
    code_match = (vendored.is_file() and runtime_hindsight.is_file() and
                  sha256(vendored) == sha256(runtime_hindsight))
    if not code_match:
        issues.append("runtime hindsight implementation differs from vendored source")
    ray_trainer = maxrl_root / "verl/trainer/ppo/ray_trainer.py"
    ray_text = (ray_trainer.read_text(encoding="utf-8", errors="replace")
                if ray_trainer.is_file() else "")
    integration_tokens = (
        "DoseMatchedLiveReplay", "ReplayReservoirCollector",
        "self.live_replay.replay_batch", "collect_batch")
    missing_tokens = [token for token in integration_tokens if token not in ray_text]
    if missing_tokens:
        issues.append(f"runtime trainer lacks E2c hooks: {missing_tokens}")

    comparators = []
    for seed in FROZEN_SEEDS:
        for arm in ("b1", "b2"):
            item, item_issues = inspect_comparator(
                runtime_root, research_root, active_log_dir, arm, seed)
            comparators.append(item)
            issues.extend(item_issues)
    reservoir, reservoir_issues = inspect_reservoir(
        runtime_root, research_root, active_log_dir)
    issues.extend(reservoir_issues)
    e2c_runs = []
    for seed in FROZEN_SEEDS:
        item, item_issues = inspect_e2c_run(
            runtime_root, research_root, active_log_dir, seed)
        e2c_runs.append(item)
        issues.extend(item_issues)

    preflight = iteration_dir / "E2C_PREFLIGHT.json"
    delivery = iteration_dir / "E2C_DELIVERY.json"
    endpoint_files = sorted(result_dir.glob("*_eval.json*"))
    missing_comparators = [
        item["run_id"] for item in comparators if item["status"] == "missing"]
    missing_e2c = [
        item["run_id"] for item in e2c_runs if item["status"] == "missing"]
    preflight_status = "missing"
    if preflight.is_file():
        preflight_status = "invalid"
        try:
            with preflight.open(encoding="utf-8") as handle:
                preflight_report = json.load(handle)
            schedule_seeds = [
                int(item["seed"])
                for item in preflight_report.get("schedule_validation", [])]
            if preflight_report.get("status") != "pass":
                issues.append("static E2c preflight does not report pass")
            elif tuple(schedule_seeds) != FROZEN_SEEDS:
                issues.append("static E2c preflight has the wrong seed set")
            elif preflight_report.get("train_data_sha256") != \
                    FROZEN_ASSET_SHA256["train.parquet"]:
                issues.append("static E2c preflight names the wrong train data")
            elif preflight_report.get("test_data_sha256") != \
                    FROZEN_ASSET_SHA256["test.parquet"]:
                issues.append("static E2c preflight names the wrong held-out data")
            elif preflight_report.get("maxrl_commit") != FROZEN_MAXRL_COMMIT:
                issues.append("static E2c preflight names the wrong MaxRL commit")
            elif preflight_report.get(
                    "base_model_revision") != FROZEN_BASE_MODEL_REVISION:
                issues.append("static E2c preflight names the wrong base model")
            elif any(
                    preflight_report.get("frozen_model_files", {}).get(
                        name, {}).get("sha256") != FROZEN_ASSET_SHA256[name]
                    for name in ("config.json", "model.safetensors",
                                 "tokenizer.json", "training_metrics.json")):
                issues.append("static E2c preflight names the wrong SFT model")
            elif reservoir.get("sha256") != preflight_report.get(
                    "reservoir_sha256"):
                issues.append("static E2c preflight names the wrong reservoir")
            elif missing_comparators or reservoir["status"] != "complete":
                issues.append("static E2c preflight exists before its inputs")
            else:
                preflight_status = "pass"
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            issues.append(f"static E2c preflight is unreadable: {error}")

    delivery_status = "missing"
    if delivery.is_file():
        delivery_status = "invalid"
        try:
            with delivery.open(encoding="utf-8") as handle:
                delivery_report = json.load(handle)
            delivery_seeds = [
                int(item["seed"])
                for item in delivery_report.get("seeds", [])]
            if (delivery_report.get("status") != "pass" or
                    not delivery_report.get("endpoint_evaluation_permitted")):
                issues.append("E2c delivery gate does not permit endpoints")
            elif tuple(delivery_seeds) != FROZEN_SEEDS:
                issues.append("E2c delivery gate has the wrong seed set")
            elif any(item.get("status") != "pass"
                     for item in delivery_report["seeds"]):
                issues.append("an E2c seed lacks passing delivery")
            elif missing_e2c:
                issues.append("E2c delivery gate exists before all replay runs")
            else:
                delivery_status = "pass"
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            issues.append(f"E2c delivery gate is unreadable: {error}")
    if endpoint_files and delivery_status != "pass":
        issues.append("held-out endpoint artifacts exist before delivery gate")

    if issues:
        next_stage = "repair_integrity_failure"
    elif missing_comparators:
        next_stage = f"train_{missing_comparators[0]}"
    elif reservoir["status"] == "missing":
        next_stage = "collect_frozen_sft_reservoir"
    elif not preflight.is_file():
        next_stage = "run_static_reservoir_and_schedule_preflight"
    elif missing_e2c:
        next_stage = f"train_{missing_e2c[0]}"
    elif not delivery.is_file():
        next_stage = "validate_all_three_delivery_audits"
    elif len(endpoint_files) < 18:
        # Nine arm/seed pairs, each with one summary and one raw JSONL file.
        next_stage = "generate_paired_heldout_endpoints"
    else:
        next_stage = "analyze_and_propagate_e2c_verdict"

    try:
        gpu_used = int(_command(
            "nvidia-smi", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits").splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError):
        gpu_used = None
    gpu_gate_pass = (gpu_used is not None and
                     gpu_used <= FROZEN_GPU_MEMORY_CEILING_MIB)
    gpu_processes = []
    try:
        process_output = _command(
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits")
        for line in process_output.splitlines():
            pid, process_name, used_memory = (
                value.strip() for value in line.split(",", 2))
            gpu_processes.append({
                "pid": int(pid),
                "process_name": process_name,
                "used_memory_mib": int(used_memory),
            })
    except (OSError, subprocess.CalledProcessError, ValueError):
        gpu_processes = []
    report = {
        "audit_kind": "outcome_blind_e2c_launch_readiness",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "integrity_status": "pass" if not issues else "fail",
        "issues": issues,
        "frozen_protocol": {
            "seeds": list(FROZEN_SEEDS),
            "steps": FROZEN_STEPS,
            "gpu_memory_ceiling_mib": FROZEN_GPU_MEMORY_CEILING_MIB,
            "maxrl_commit": FROZEN_MAXRL_COMMIT,
            "base_model_revision": FROZEN_BASE_MODEL_REVISION,
        },
        "assets": asset_report,
        "code_manifest": code_manifest,
        "runtime": {
            "maxrl_root": str(maxrl_root),
            "maxrl_commit": actual_commit,
            "vendored_runtime_hindsight_match": code_match,
            "trainer_integration_tokens_missing": missing_tokens,
        },
        "comparators": comparators,
        "reservoir": reservoir,
        "static_preflight": {
            "path": str(preflight),
            "status": preflight_status,
        },
        "e2c_runs": e2c_runs,
        "delivery_gate": {
            "path": str(delivery),
            "status": delivery_status,
        },
        "heldout_artifact_count": len(endpoint_files),
        "heldout_artifacts_inspected": False,
        "next_stage": next_stage,
        "gpu": {
            "memory_used_mib": gpu_used,
            "ceiling_mib": FROZEN_GPU_MEMORY_CEILING_MIB,
            "gate_pass": gpu_gate_pass,
            "compute_processes": gpu_processes,
        },
        "launch_authorized_now": not issues and gpu_gate_pass,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-root",
        default="/data/robotixx/curriculum-maxrl-runtime")
    parser.add_argument("--research-root", default=str(Path.cwd()))
    parser.add_argument(
        "--iteration-dir",
        default="autoresearch/iterate-260810-2240")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_readiness(
        Path(args.runtime_root), Path(args.research_root),
        Path(args.iteration_dir))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["integrity_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
