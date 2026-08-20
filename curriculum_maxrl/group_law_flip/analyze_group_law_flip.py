#!/usr/bin/env python3
"""Fail-closed analysis for the frozen GROUP-LAW-FLIP v1 campaign.

The analyzer accepts one attempt directory containing exactly 48 completed
paired seed blocks.  It verifies directory shape, hashes, completion receipts,
the shared warmstart, source identity, run configuration, all 500 result
records, and all 24,000 telemetry records before calculating an endpoint.

The command-line path is single-use: an exclusive claim file is created only
after structural validation and treatment-delivery accounting succeed.  No
endpoint value is printed on progress or refusal paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL = "group_law_flip_v1"
SCHEMA = "curriculum-maxrl/group-law-flip/analysis/v1"
CAMPAIGN_ID = "group-law-flip-v1-20260820-001"
ATTEMPT_ID = "attempt-001"
ARMS = ("plugin", "grouplaw")
EXPECTED_SEEDS = tuple(range(3001, 3049))
EXPECTED_UPDATES = tuple(range(25, 251, 25))
EXPECTED_LEVELS = tuple(str(level) for level in range(13))
EXPECTED_ROLLOUTS = 32
EXPECTED_STEPS = 250
EXPECTED_EVAL_EVERY = 25
EXPECTED_EVAL_TASKS_PER_LEVEL = 32
EXPECTED_TASKS_PER_STEP = 8
EXPECTED_SFT_STEPS = 600
EXPECTED_EVAL_SAMPLES = 8
EXPECTED_LR = 1e-4
EXPECTED_D_MODEL = 128
EXPECTED_N_LAYERS = 6
EXPECTED_TEACHER_POWER = 1.0
EXPECTED_POSTERIOR_PRIOR_P0 = 0.5
EXPECTED_POSTERIOR_PRIOR_MASS = 2.0 / EXPECTED_ROLLOUTS
EXPECTED_POSTERIOR_DECAY = 0.7
EXPECTED_TEACHER_FLOOR = 0.15
BOOTSTRAP_RESAMPLES = 20_000
RANDOM_SEED = 20_260_820
SESOI = 0.005
DELIVERY_TV_THRESHOLD = 0.05
MAX_EXACT_SIGN_FLIP_N = 48

_RESULT_RE = re.compile(r"^groupflip_(plugin|grouplaw)_s(\d+)\.jsonl$")
_TELEMETRY_RE = re.compile(
    r"^groupflip_(plugin|grouplaw)_s(\d+)\.telemetry\.jsonl$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_LINE_RE = re.compile(r"^([0-9a-f]{64})  (\./[^\n]+)$")


class AnalysisError(ValueError):
    """Raised when evidence violates the frozen analysis contract."""


@dataclass(frozen=True)
class Run:
    path: Path
    config: Mapping[str, Any]
    baseline_by_level: Mapping[str, float]
    updates_by_level: Mapping[int, Mapping[str, float]]

    @property
    def arm(self) -> str:
        return str(self.config["arm"])

    @property
    def seed(self) -> int:
        return int(self.config["seed"])


@dataclass(frozen=True)
class Telemetry:
    path: Path
    arm: str
    seed: int
    visit_counts: np.ndarray
    level_counts: Mapping[int, tuple[int, ...]]


@dataclass(frozen=True)
class ValidatedCampaign:
    root: Path
    source_manifest_sha256: str
    runs: Mapping[tuple[int, str], Run]
    telemetry: Mapping[tuple[int, str], Telemetry]
    block_tv: Mapping[int, float]

    @property
    def mean_delivery_tv(self) -> float:
        return float(np.mean([self.block_tv[seed] for seed in EXPECTED_SEEDS]))


def _reject_json_constant(value: str) -> None:
    raise AnalysisError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnalysisError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(raw: str, where: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, AnalysisError) as exc:
        raise AnalysisError(f"{where}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{where}: JSON object required")
    return value


def _load_json_file(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AnalysisError(f"missing or symlinked JSON file: {path}")
    return _load_json(path.read_text(encoding="utf-8"), str(path))


def _jsonl(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    if not path.is_file() or path.is_symlink():
        raise AnalysisError(f"missing or symlinked JSONL file: {path}")
    records: list[tuple[int, Mapping[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if raw.strip():
                records.append((line_number, _load_json(raw, f"{path.name}:{line_number}")))
    if not records:
        raise AnalysisError(f"{path.name}: empty JSONL")
    return records


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(obj: Mapping[str, Any], key: str, expected: int | None = None) -> int:
    value = obj.get(key)
    if not _is_int(value):
        raise AnalysisError(f"field {key!r} must be an integer")
    if expected is not None and value != expected:
        raise AnalysisError(f"field {key!r} must equal {expected}, got {value}")
    return int(value)


def _require_number(obj: Mapping[str, Any], key: str, expected: float) -> float:
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"field {key!r} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric != expected:
        raise AnalysisError(f"field {key!r} must equal {expected}, got {value}")
    return numeric


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AnalysisError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(config: Mapping[str, Any], path: Path) -> None:
    if config.get("record_type") != "config" or config.get("protocol") != PROTOCOL:
        raise AnalysisError(f"{path.name}: malformed {PROTOCOL} config")
    if config.get("campaign") != CAMPAIGN_ID:
        raise AnalysisError(f"{path.name}: campaign must equal {CAMPAIGN_ID!r}")
    source = _require_sha(config.get("source_manifest"), "source_manifest")
    if config.get("estimator") != "maxrl" or config.get("score_estimator") != "maxrl":
        raise AnalysisError(f"{path.name}: estimator fields must both be 'maxrl'")

    arm = config.get("arm")
    if arm not in ARMS:
        raise AnalysisError(f"{path.name}: arm must be one of {ARMS}")
    seed = _require_int(config, "seed")
    if seed not in EXPECTED_SEEDS:
        raise AnalysisError(f"{path.name}: seed {seed} is outside the frozen blocks")
    match = _RESULT_RE.fullmatch(path.name)
    if match is None or match.group(1) != arm or int(match.group(2)) != seed:
        raise AnalysisError(f"{path.name}: filename and config arm/seed disagree")

    seeds = config.get("seeds")
    if not isinstance(seeds, dict):
        raise AnalysisError(f"{path.name}: seeds must be an object")
    expected_seeds = {
        "base": seed,
        "sft": seed,
        "rl": seed,
        "teacher": seed + 77,
        "eval_tasks": 202_608_130 + seed,
        "eval_samples": 302_608_130 + seed,
    }
    for name, expected in expected_seeds.items():
        if seeds.get(name) != expected or not _is_int(seeds.get(name)):
            raise AnalysisError(f"{path.name}: seeds.{name} must equal {expected}")

    _require_int(config, "rollouts", EXPECTED_ROLLOUTS)
    _require_int(config, "steps", EXPECTED_STEPS)
    _require_int(config, "eval_every", EXPECTED_EVAL_EVERY)
    _require_int(config, "eval_tasks_per_level", EXPECTED_EVAL_TASKS_PER_LEVEL)
    _require_int(config, "tasks_per_step", EXPECTED_TASKS_PER_STEP)
    _require_int(config, "sft_steps", EXPECTED_SFT_STEPS)
    _require_int(config, "eval_samples", EXPECTED_EVAL_SAMPLES)
    _require_int(config, "planned_rl_eval_count", len(EXPECTED_UPDATES))
    _require_int(config, "d_model", EXPECTED_D_MODEL)
    _require_int(config, "n_layers", EXPECTED_N_LAYERS)
    _require_number(config, "lr", EXPECTED_LR)
    _require_number(config, "teacher_power", EXPECTED_TEACHER_POWER)
    _require_number(config, "posterior_prior_p0", EXPECTED_POSTERIOR_PRIOR_P0)
    _require_number(config, "posterior_prior_mass", EXPECTED_POSTERIOR_PRIOR_MASS)
    _require_number(config, "posterior_decay", EXPECTED_POSTERIOR_DECAY)
    _require_number(config, "teacher_floor", EXPECTED_TEACHER_FLOOR)
    for key in ("hindsight", "hindsight_dense", "hindsight_to_teacher"):
        if config.get(key) is not False:
            raise AnalysisError(f"{path.name}: {key} must be false")
    if config.get("posterior_family") != "count_law_moments":
        raise AnalysisError(f"{path.name}: wrong posterior_family")
    if config.get("posterior_sampling") != "posterior_mean":
        raise AnalysisError(f"{path.name}: wrong posterior_sampling")
    _require_sha(config.get("sft_checkpoint_sha256"), "sft_checkpoint_sha256")
    if not isinstance(config.get("sft_checkpoint"), str) or not config["sft_checkpoint"]:
        raise AnalysisError(f"{path.name}: sft_checkpoint must be nonempty")

    if arm == "plugin":
        expected = ("group_law_plugin", "iid_plugin_from_count_law_mean", EXPECTED_ROLLOUTS)
    else:
        expected = ("group_law_activity", "group_law_activity", None)
    observed = (config.get("teacher"), config.get("score_family"), config.get("effective_exponent"))
    if observed != expected:
        raise AnalysisError(
            f"{path.name}: teacher/score/exponent must equal {expected!r}, got {observed!r}"
        )
    # Keep the validated value live; this line also makes accidental relaxation
    # of source_manifest validation conspicuous to coverage tools.
    assert source == config["source_manifest"]


def _pass8_by_level(record: Mapping[str, Any], where: str) -> dict[str, float]:
    passk = record.get("passk")
    if not isinstance(passk, dict) or set(passk) != set(EXPECTED_LEVELS):
        raise AnalysisError(f"{where}: passk must contain exactly levels 0..12")
    values: dict[str, float] = {}
    for level in EXPECTED_LEVELS:
        per_k = passk[level]
        if not isinstance(per_k, dict) or set(per_k) != {"1", "8"}:
            raise AnalysisError(f"{where}: level {level} must contain exactly pass@1,8")
        value = per_k["8"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AnalysisError(f"{where}: level {level} pass@8 must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise AnalysisError(f"{where}: level {level} pass@8 outside [0,1]")
        values[level] = numeric
    return values


def load_run(path: str | os.PathLike[str]) -> Run:
    run_path = Path(path)
    records = _jsonl(run_path)
    configs = [(line, row) for line, row in records if row.get("record_type") == "config"]
    if len(configs) != 1:
        raise AnalysisError(f"{run_path.name}: exactly one config record required")
    config = configs[0][1]
    _validate_config(config, run_path)

    evaluations: list[tuple[int, Mapping[str, Any]]] = []
    for line, row in records:
        if row.get("record_type") == "evaluation":
            if row.get("protocol") != PROTOCOL:
                raise AnalysisError(f"{run_path.name}:{line}: wrong evaluation protocol")
            evaluations.append((line, row))
        elif row.get("record_type") != "config":
            raise AnalysisError(f"{run_path.name}:{line}: unknown record_type")
    baselines = [(line, row) for line, row in evaluations if row.get("phase") == "post_sft"]
    if len(baselines) != 1:
        raise AnalysisError(f"{run_path.name}: exactly one post-SFT baseline required")
    baseline_line, baseline = baselines[0]
    if baseline.get("completed_updates") != 0 or baseline.get("final", False) is not False:
        raise AnalysisError(f"{run_path.name}:{baseline_line}: malformed post-SFT baseline")
    baseline_values = _pass8_by_level(baseline, f"{run_path.name}:{baseline_line}")

    updates: dict[int, Mapping[str, float]] = {}
    for line, row in evaluations:
        if row is baseline:
            continue
        if row.get("phase") != "rl" or not _is_int(row.get("completed_updates")):
            raise AnalysisError(f"{run_path.name}:{line}: malformed RL evaluation")
        update = int(row["completed_updates"])
        if update in updates:
            raise AnalysisError(f"{run_path.name}: duplicate timepoint {update}")
        final = row.get("final", False)
        if not isinstance(final, bool) or final is not (update == EXPECTED_STEPS):
            raise AnalysisError(f"{run_path.name}:{line}: final flag violates full-budget contract")
        updates[update] = _pass8_by_level(row, f"{run_path.name}:{line}")
    if set(updates) != set(EXPECTED_UPDATES):
        missing = sorted(set(EXPECTED_UPDATES) - set(updates))
        extra = sorted(set(updates) - set(EXPECTED_UPDATES))
        raise AnalysisError(f"{run_path.name}: timepoints mismatch; missing={missing}, extra={extra}")
    return Run(run_path, dict(config), baseline_values, updates)


def load_telemetry(path: str | os.PathLike[str]) -> Telemetry:
    telemetry_path = Path(path)
    match = _TELEMETRY_RE.fullmatch(telemetry_path.name)
    if match is None:
        raise AnalysisError(f"unexpected telemetry filename: {telemetry_path.name}")
    arm, seed = match.group(1), int(match.group(2))
    if seed not in EXPECTED_SEEDS:
        raise AnalysisError(f"{telemetry_path.name}: seed outside frozen blocks")
    records = _jsonl(telemetry_path)
    if len(records) != EXPECTED_STEPS:
        raise AnalysisError(f"{telemetry_path.name}: exactly {EXPECTED_STEPS} records required")

    visits = np.zeros(len(EXPECTED_LEVELS), dtype=np.int64)
    by_level: dict[int, list[int]] = defaultdict(list)
    expected_optimizer_steps = 0
    for expected_update, (line, row) in enumerate(records, 1):
        where = f"{telemetry_path.name}:{line}"
        if row.get("record_type") != "telemetry" or row.get("protocol") != PROTOCOL:
            raise AnalysisError(f"{where}: malformed telemetry record")
        if row.get("completed_updates") != expected_update:
            raise AnalysisError(f"{where}: updates must be exactly 1..{EXPECTED_STEPS}")
        levels, counts, masses = (
            row.get("selected_levels"), row.get("group_k"), row.get("coefficient_mass")
        )
        if not all(isinstance(value, list) for value in (levels, counts, masses)):
            raise AnalysisError(f"{where}: levels/counts/masses must be lists")
        if not all(len(value) == EXPECTED_TASKS_PER_STEP for value in (levels, counts, masses)):
            raise AnalysisError(f"{where}: exactly {EXPECTED_TASKS_PER_STEP} groups required")
        expected_masses: list[float] = []
        dead = 0
        for level, count, mass in zip(levels, counts, masses, strict=True):
            if not _is_int(level) or not 0 <= level < len(EXPECTED_LEVELS):
                raise AnalysisError(f"{where}: invalid selected level {level!r}")
            if not _is_int(count) or not 0 <= count <= EXPECTED_ROLLOUTS:
                raise AnalysisError(f"{where}: invalid group count {count!r}")
            if isinstance(mass, bool) or not isinstance(mass, (int, float)) or not math.isfinite(float(mass)):
                raise AnalysisError(f"{where}: invalid coefficient mass {mass!r}")
            expected_mass = 0.0 if count == 0 else 2.0 * (1.0 - count / EXPECTED_ROLLOUTS)
            if not math.isclose(float(mass), expected_mass, rel_tol=0.0, abs_tol=1e-12):
                raise AnalysisError(f"{where}: coefficient mass disagrees with MaxRL count law")
            expected_masses.append(expected_mass)
            dead += int(expected_mass == 0.0)
            visits[level] += 1
            by_level[level].append(count)
        total = row.get("coefficient_mass_total")
        if isinstance(total, bool) or not isinstance(total, (int, float)):
            raise AnalysisError(f"{where}: coefficient_mass_total must be numeric")
        if not math.isclose(float(total), sum(expected_masses), rel_tol=0.0, abs_tol=1e-12):
            raise AnalysisError(f"{where}: coefficient_mass_total mismatch")
        if row.get("dead_groups") != dead or not _is_int(row.get("dead_groups")):
            raise AnalysisError(f"{where}: dead_groups mismatch")
        applied = dead < EXPECTED_TASKS_PER_STEP
        if row.get("optimizer_step_applied") is not applied:
            raise AnalysisError(f"{where}: optimizer_step_applied mismatch")
        expected_optimizer_steps += int(applied)
        if row.get("optimizer_step") != expected_optimizer_steps:
            raise AnalysisError(f"{where}: optimizer_step counter mismatch")
    if int(visits.sum()) != EXPECTED_STEPS * EXPECTED_TASKS_PER_STEP:
        raise AnalysisError(f"{telemetry_path.name}: visit total mismatch")
    return Telemetry(
        telemetry_path,
        arm,
        seed,
        visits,
        {level: tuple(values) for level, values in by_level.items()},
    )


def _validate_manifest(block: Path) -> None:
    manifest = block / "SHA256SUMS"
    if not manifest.is_file() or manifest.is_symlink():
        raise AnalysisError(f"{block.name}: missing SHA256SUMS")
    actual_files: set[str] = set()
    for path in block.rglob("*"):
        if path.is_symlink():
            raise AnalysisError(f"{block.name}: symlinks are forbidden: {path}")
        if path.is_file() and path.name not in {"SHA256SUMS", "COMPLETE"}:
            actual_files.add("./" + path.relative_to(block).as_posix())
    entries: dict[str, str] = {}
    lines = manifest.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        match = _MANIFEST_LINE_RE.fullmatch(line)
        if match is None:
            raise AnalysisError(f"{block.name}:SHA256SUMS:{line_number}: malformed line")
        digest, relative = match.groups()
        if relative in entries:
            raise AnalysisError(f"{block.name}: duplicate manifest entry {relative}")
        if relative.startswith("./../") or "/../" in relative or relative == "./.":
            raise AnalysisError(f"{block.name}: unsafe manifest path {relative}")
        entries[relative] = digest
    if set(entries) != actual_files:
        missing = sorted(actual_files - set(entries))
        extra = sorted(set(entries) - actual_files)
        raise AnalysisError(f"{block.name}: manifest coverage mismatch; missing={missing}, extra={extra}")
    for relative, expected in entries.items():
        path = block / relative[2:]
        if _sha256(path) != expected:
            raise AnalysisError(f"{block.name}: hash mismatch for {relative}")


def _validate_receipt(
    receipt: Mapping[str, Any], block: Path, arm: str, seed: int, run: Run
) -> None:
    expected_paths = {
        "result": f"results/groupflip_{arm}_s{seed}.jsonl",
        "telemetry": f"telemetry/groupflip_{arm}_s{seed}.telemetry.jsonl",
        "checkpoint": f"checkpoints/groupflip_{arm}_s{seed}.pt",
    }
    expected_scalars = {
        "schema": "curriculum-maxrl/group-law-flip/arm-receipt/v1",
        "protocol": PROTOCOL,
        "campaign": CAMPAIGN_ID,
        "attempt": ATTEMPT_ID,
        "arm": arm,
        "seed": seed,
        "completed_updates": EXPECTED_STEPS,
        "source_manifest_sha256": run.config["source_manifest"],
        "sft_checkpoint_sha256": run.config["sft_checkpoint_sha256"],
    }
    for key, expected in expected_scalars.items():
        if receipt.get(key) != expected:
            raise AnalysisError(f"{block.name}: {arm} receipt field {key!r} mismatch")
    paths = receipt.get("paths")
    hashes = receipt.get("sha256")
    if paths != expected_paths or not isinstance(hashes, dict) or set(hashes) != set(expected_paths):
        raise AnalysisError(f"{block.name}: {arm} receipt paths/hashes mismatch")
    for key, relative in expected_paths.items():
        expected_hash = _require_sha(hashes.get(key), f"{arm} receipt sha256.{key}")
        if _sha256(block / relative) != expected_hash:
            raise AnalysisError(f"{block.name}: {arm} receipt hash mismatch for {key}")


def _validate_block(block: Path, seed: int) -> tuple[dict[str, Run], dict[str, Telemetry], str]:
    if not block.is_dir() or block.is_symlink() or block.name != f"seed-{seed}":
        raise AnalysisError(f"missing or malformed seed block {seed}")
    expected_dirs = {"results", "telemetry", "warmstarts", "checkpoints", "meta"}
    actual_dirs = {path.name for path in block.iterdir() if path.is_dir()}
    if actual_dirs != expected_dirs:
        raise AnalysisError(f"{block.name}: directory set mismatch")
    unexpected_top = {
        path.name for path in block.iterdir()
        if path.is_file() and path.name not in {"SHA256SUMS", "COMPLETE"}
    }
    if unexpected_top:
        raise AnalysisError(f"{block.name}: unexpected top-level files {sorted(unexpected_top)}")
    _validate_manifest(block)

    complete = _load_json_file(block / "COMPLETE")
    expected_complete = {
        "schema": "curriculum-maxrl/group-law-flip/block-complete/v1",
        "protocol": PROTOCOL,
        "campaign": CAMPAIGN_ID,
        "attempt": ATTEMPT_ID,
        "seed": seed,
        "completed_arms": list(ARMS),
    }
    for key, expected in expected_complete.items():
        if complete.get(key) != expected:
            raise AnalysisError(f"{block.name}: COMPLETE field {key!r} mismatch")

    runs: dict[str, Run] = {}
    telemetry: dict[str, Telemetry] = {}
    for arm in ARMS:
        run = load_run(block / "results" / f"groupflip_{arm}_s{seed}.jsonl")
        telem = load_telemetry(
            block / "telemetry" / f"groupflip_{arm}_s{seed}.telemetry.jsonl"
        )
        if (run.arm, run.seed) != (telem.arm, telem.seed):
            raise AnalysisError(f"{block.name}: result/telemetry identity mismatch")
        receipt = _load_json_file(block / "meta" / f"{arm}.DONE.json")
        _validate_receipt(receipt, block, arm, seed, run)
        runs[arm], telemetry[arm] = run, telem

    warmstart = block / "warmstarts" / f"seed-{seed}-sft.pt"
    if not warmstart.is_file() or warmstart.is_symlink():
        raise AnalysisError(f"{block.name}: missing shared warmstart")
    warm_hash = _sha256(warmstart)
    config_hashes = {run.config["sft_checkpoint_sha256"] for run in runs.values()}
    if config_hashes != {warm_hash}:
        raise AnalysisError(f"{block.name}: arms do not share the recorded warmstart")
    source_hashes = {str(run.config["source_manifest"]) for run in runs.values()}
    if len(source_hashes) != 1:
        raise AnalysisError(f"{block.name}: arm source manifests disagree")
    source_hash = next(iter(source_hashes))
    source_copy = block / "meta" / "SOURCE_SHA256SUMS"
    if _sha256(source_copy) != source_hash:
        raise AnalysisError(f"{block.name}: copied source manifest hash mismatch")
    return runs, telemetry, source_hash


def validate_campaign(root: str | os.PathLike[str]) -> ValidatedCampaign:
    campaign_root = Path(root).resolve()
    if not campaign_root.is_dir() or campaign_root.is_symlink():
        raise AnalysisError(f"campaign root is not a directory: {campaign_root}")
    expected_names = {f"seed-{seed}" for seed in EXPECTED_SEEDS}
    actual_names = {path.name for path in campaign_root.iterdir() if path.is_dir()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise AnalysisError(f"campaign block set mismatch; missing={missing}, extra={extra}")
    unexpected_files = {
        path.name for path in campaign_root.iterdir()
        if path.is_file() and path.name != ".GROUP_LAW_FLIP_ANALYSIS_CLAIM.json"
    }
    if unexpected_files:
        raise AnalysisError(f"unexpected campaign-root files: {sorted(unexpected_files)}")

    all_runs: dict[tuple[int, str], Run] = {}
    all_telemetry: dict[tuple[int, str], Telemetry] = {}
    source_hash: str | None = None
    block_tv: dict[int, float] = {}
    for seed in EXPECTED_SEEDS:
        runs, telemetry, block_source = _validate_block(campaign_root / f"seed-{seed}", seed)
        if source_hash is None:
            source_hash = block_source
        elif source_hash != block_source:
            raise AnalysisError("source manifest mismatch across seed blocks")
        for arm in ARMS:
            all_runs[(seed, arm)] = runs[arm]
            all_telemetry[(seed, arm)] = telemetry[arm]
        p = telemetry["plugin"].visit_counts.astype(float)
        q = telemetry["grouplaw"].visit_counts.astype(float)
        p /= p.sum()
        q /= q.sum()
        block_tv[seed] = float(0.5 * np.abs(p - q).sum())

    mean_tv = float(np.mean(list(block_tv.values())))
    if not math.isfinite(mean_tv):
        raise AnalysisError("non-finite treatment-delivery statistic")
    assert source_hash is not None
    return ValidatedCampaign(
        campaign_root, source_hash, all_runs, all_telemetry, block_tv
    )


def exact_sign_flip_p(differences: Sequence[float]) -> float:
    """Exact two-sided paired randomization p-value via meet-in-the-middle."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise AnalysisError("paired differences must be a nonempty finite vector")
    if values.size > MAX_EXACT_SIGN_FLIP_N:
        raise AnalysisError(
            f"exact sign-flip test supports at most {MAX_EXACT_SIGN_FLIP_N} pairs"
        )

    def signed_sums(part: np.ndarray) -> np.ndarray:
        sums = np.zeros(1, dtype=float)
        for value in part:
            sums = np.concatenate((sums - value, sums + value))
        return sums

    split = values.size // 2
    left = signed_sums(values[:split])
    right = np.sort(signed_sums(values[split:]))
    observed = abs(float(values.sum()))
    tolerance = max(1e-15, 1e-14 * max(1.0, observed))
    threshold = max(0.0, observed - tolerance)
    if threshold == 0.0:
        return 1.0
    lower = int(np.sum(np.searchsorted(right, -threshold - left, side="right"), dtype=np.int64))
    upper_indices = np.searchsorted(right, threshold - left, side="left")
    upper = int(np.sum(right.size - upper_indices, dtype=np.int64))
    return float((lower + upper) / (1 << int(values.size)))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    rx, ry = _rankdata(x), _rankdata(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _cov_auc_by_level(run: Run) -> dict[str, float]:
    return {
        level: float(
            np.mean([run.updates_by_level[update][level] for update in EXPECTED_UPDATES])
            - run.baseline_by_level[level]
        )
        for level in EXPECTED_LEVELS
    }


def _cov_auc(run: Run) -> float:
    by_level = _cov_auc_by_level(run)
    return float(np.mean(list(by_level.values())))


def _descriptive_levels(campaign: ValidatedCampaign) -> dict[str, Any]:
    pooled: dict[int, list[int]] = defaultdict(list)
    coverage_differences: dict[int, list[float]] = defaultdict(list)
    for seed in EXPECTED_SEEDS:
        plugin_levels = _cov_auc_by_level(campaign.runs[(seed, "plugin")])
        grouplaw_levels = _cov_auc_by_level(campaign.runs[(seed, "grouplaw")])
        for level in range(len(EXPECTED_LEVELS)):
            coverage_differences[level].append(
                grouplaw_levels[str(level)] - plugin_levels[str(level)]
            )
        for arm in ARMS:
            for level, counts in campaign.telemetry[(seed, arm)].level_counts.items():
                pooled[level].extend(counts)

    rows: dict[str, Any] = {}
    gaps, covs = [], []
    for level in range(len(EXPECTED_LEVELS)):
        counts = np.asarray(pooled[level], dtype=float)
        if counts.size == 0:
            gap = None
        else:
            p_bar = float(counts.mean() / EXPECTED_ROLLOUTS)
            p_zero = float(np.mean(counts == 0))
            gap = float(2.0 * (p_zero - (1.0 - p_bar) ** EXPECTED_ROLLOUTS))
            gaps.append(gap)
            covs.append(float(np.mean(coverage_differences[level])))
        rows[str(level)] = {
            "n_groups_pooled_across_arms_and_blocks": int(counts.size),
            "measured_plugin_minus_group_law_gap": gap,
            "mean_cov_auc_difference_grouplaw_minus_plugin": float(
                np.mean(coverage_differences[level])
            ),
        }
    correlation = _spearman(np.asarray(gaps), np.asarray(covs)) if gaps else None
    return {
        "status": "descriptive_only",
        "pooling": "all observed P0 groups across both arms and all seed blocks",
        "per_level": rows,
        "spearman_gap_vs_coverage_difference": correlation,
    }


def analyze_validated(campaign: ValidatedCampaign) -> dict[str, Any]:
    """Compute the endpoint only from an already fully validated campaign."""
    cell_metrics = {
        (seed, arm): _cov_auc(campaign.runs[(seed, arm)])
        for seed in EXPECTED_SEEDS
        for arm in ARMS
    }
    differences = np.asarray(
        [cell_metrics[(seed, "grouplaw")] - cell_metrics[(seed, "plugin")] for seed in EXPECTED_SEEDS],
        dtype=float,
    )
    p_value = exact_sign_flip_p(differences)
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.integers(0, differences.size, size=(BOOTSTRAP_RESAMPLES, differences.size))
    boot_means = np.mean(differences[indices], axis=1)
    lower, upper = [float(value) for value in np.percentile(boot_means, [2.5, 97.5])]
    mean = float(differences.mean())
    delivery_passed = campaign.mean_delivery_tv >= DELIVERY_TV_THRESHOLD
    supported = delivery_passed and mean >= SESOI and lower > 0.0 and p_value <= 0.05
    practically_ruled_out = delivery_passed and not supported and upper < SESOI
    if not delivery_passed:
        decision = "treatment_not_delivered"
    elif supported:
        decision = "supported"
    elif practically_ruled_out:
        decision = "practically_ruled_out"
    else:
        decision = "inconclusive"

    cells: dict[str, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        cells[str(seed)] = {}
        for arm in ARMS:
            run = campaign.runs[(seed, arm)]
            cells[str(seed)][arm] = {
                "input": str(run.path),
                "sft_checkpoint_sha256": run.config["sft_checkpoint_sha256"],
                "post_sft_cov8": float(np.mean(list(run.baseline_by_level.values()))),
                "cov_auc_delta": cell_metrics[(seed, arm)],
            }

    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "campaign": CAMPAIGN_ID,
        "attempt": ATTEMPT_ID,
        "source_manifest_sha256": campaign.source_manifest_sha256,
        "complete": True,
        "seed_blocks": list(EXPECTED_SEEDS),
        "arms": list(ARMS),
        "treatment_delivery": {
            "metric": "per-block TV between empirical full-run level-visit distributions",
            "threshold": DELIVERY_TV_THRESHOLD,
            "mean_tv": campaign.mean_delivery_tv,
            "per_seed_tv": {str(seed): campaign.block_tv[seed] for seed in EXPECTED_SEEDS},
            "passed": delivery_passed,
        },
        "endpoint": {
            "name": "cov_auc_delta",
            "evaluation_completed_updates": list(EXPECTED_UPDATES),
            "definition": "mean pass@8 coverage over ten RL evaluations minus post-SFT coverage",
        },
        "analysis": {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "random_seed": RANDOM_SEED,
            "bootstrap_interval": "paired percentile 95% (NumPy linear percentile)",
            "randomization_test": "exact two-sided paired sign-flip test on the mean",
            "multiplicity": "one preregistered primary contrast; no adjustment",
            "support_rule": "delivery passes, mean>=0.005, CI lower>0, exact p<=0.05",
            "practically_ruled_out_rule": "delivery passes and CI upper<0.005",
        },
        "cells": cells,
        "primary_grouplaw_minus_plugin": {
            "per_seed_difference": {
                str(seed): float(value)
                for seed, value in zip(EXPECTED_SEEDS, differences, strict=True)
            },
            "mean": mean,
            "bootstrap_ci_95": [lower, upper],
            "sign_flip_p_two_sided_exact": p_value,
            "positive_pairs": int(np.sum(differences > 0.0)),
            "negative_pairs": int(np.sum(differences < 0.0)),
            "zero_pairs": int(np.sum(differences == 0.0)),
            "sesoi": SESOI,
            "decision": decision,
        },
        "descriptive_secondary": _descriptive_levels(campaign),
    }


def analyze_campaign(root: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate the complete campaign, then calculate its frozen analysis."""
    return analyze_validated(validate_campaign(root))


def _atomic_write_json(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary_name = handle.name
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _claim_analysis(root: Path, output: Path) -> Path:
    claim = root / ".GROUP_LAW_FLIP_ANALYSIS_CLAIM.json"
    payload = json.dumps(
        {
            "schema": "curriculum-maxrl/group-law-flip/analysis-claim/v1",
            "campaign": CAMPAIGN_ID,
            "attempt": ATTEMPT_ID,
            "output": str(output.resolve()),
        },
        indent=2,
        sort_keys=True,
    ) + "\n"
    try:
        descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    except FileExistsError as exc:
        raise AnalysisError(f"single-use analysis already claimed: {claim}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return claim


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", help="attempt directory containing seed-3001..3048")
    parser.add_argument("--output", required=True, help="new JSON output path")
    args = parser.parse_args(argv)
    root, output = Path(args.campaign_root).resolve(), Path(args.output).resolve()
    try:
        if output.exists() or output.is_symlink():
            raise AnalysisError(f"refusing to overwrite output: {output}")
        if root == output or root in output.parents:
            raise AnalysisError("analysis output must be outside the immutable campaign root")
        validated = validate_campaign(root)
        if validated.mean_delivery_tv < DELIVERY_TV_THRESHOLD:
            # This progress message names only the randomized treatment, never
            # an evaluation endpoint.
            print("campaign validated; treatment-delivery gate did not pass", flush=True)
        else:
            print("campaign validated; treatment-delivery gate passed", flush=True)
        _claim_analysis(root, output)
        result = analyze_validated(validated)
        _atomic_write_json(output, result)
        print(f"single-use GROUP-LAW-FLIP analysis complete -> {output}")
        return 0
    except AnalysisError as exc:
        print(f"GROUP-LAW-FLIP analysis refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
