"""Seed-level E2c endpoint contrasts, gated on valid treatment delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from curriculum_maxrl.countdown.analyze_e2c import summarize_displacement
from curriculum_maxrl.countdown.e2c_protocol import (
    FROZEN_ASSET_SHA256,
    FROZEN_SEEDS,
    require_frozen_seeds,
)


ARMS = ("b1", "b2", "e2c")
METRICS = ("mean@16", "pass@16")
EXPECTED_K = 16
EXPECTED_TIERS = tuple(f"countdown_tier{tier}" for tier in range(3))
EXPECTED_TASKS_PER_TIER = 128
EXPECTED_MAX_NEW_TOKENS = 128


def _read_raw_outcomes(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_endpoint_result(path: Path) -> tuple[dict[str, float], dict]:
    """Recompute the registered endpoints from retained binary task outcomes."""
    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    k = int(result.get("k", -1))
    if k != EXPECTED_K:
        raise ValueError(f"{path}: k={k}, expected {EXPECTED_K}")

    raw_value = result.get("raw_outcomes")
    if not raw_value:
        raise ValueError(f"{path}: missing raw_outcomes")
    raw_path = Path(raw_value)
    if not raw_path.is_absolute():
        raw_path = (path.parent / raw_path).resolve()
    if not raw_path.is_file():
        raise ValueError(f"{path}: raw outcomes do not exist: {raw_path}")

    by_tier: dict[str, list[list[int]]] = defaultdict(list)
    task_manifest = []
    for row_index, row in enumerate(_read_raw_outcomes(raw_path), start=1):
        tier = str(row.get("data_source"))
        if tier not in EXPECTED_TIERS:
            raise ValueError(f"{raw_path}:{row_index}: unknown tier {tier!r}")
        rewards = row.get("rewards")
        if not isinstance(rewards, list) or len(rewards) != k:
            raise ValueError(
                f"{raw_path}:{row_index}: expected {k} binary rewards")
        if any(value not in (0, 1, False, True) for value in rewards):
            raise ValueError(f"{raw_path}:{row_index}: rewards are not binary")
        truth = row.get("ground_truth")
        if not isinstance(truth, dict):
            raise ValueError(f"{raw_path}:{row_index}: missing ground truth")
        try:
            identity = (
                tier,
                int(truth["target"]),
                tuple(sorted(int(value) for value in truth["numbers"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{raw_path}:{row_index}: invalid ground truth") from error
        task_manifest.append(identity)
        by_tier[tier].append([int(value) for value in rewards])

    if set(by_tier) != set(EXPECTED_TIERS):
        raise ValueError(f"{raw_path}: missing evaluation tier")
    if len(set(task_manifest)) != len(task_manifest):
        raise ValueError(f"{raw_path}: duplicate held-out task identity")
    recomputed = {}
    for tier in EXPECTED_TIERS:
        rows = by_tier[tier]
        if len(rows) != EXPECTED_TASKS_PER_TIER:
            raise ValueError(
                f"{raw_path}: {tier} has {len(rows)} tasks, expected "
                f"{EXPECTED_TASKS_PER_TIER}")
        values = [value for rewards in rows for value in rewards]
        recomputed[tier] = {
            "mean@16": sum(values) / len(values),
            # With n=k=16, standard without-replacement pass@16 is exactly
            # the indicator that the retained 16-sample set has a success.
            "pass@16": sum(any(rewards) for rewards in rows) / len(rows),
        }
        reported = result.get("tiers", {}).get(tier, {})
        for metric in METRICS:
            if metric not in reported or not math.isclose(
                    float(reported[metric]), recomputed[tier][metric],
                    rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"{path}: {tier} {metric} does not match raw outcomes")

    data_sha256 = str(result.get("data_sha256", ""))
    if data_sha256 != FROZEN_ASSET_SHA256["test.parquet"]:
        raise ValueError(f"{path}: held-out dataset checksum is not frozen E2c")
    max_new_tokens = int(result["max_new_tokens"])
    if max_new_tokens != EXPECTED_MAX_NEW_TOKENS:
        raise ValueError(
            f"{path}: max_new_tokens={max_new_tokens}, expected "
            f"{EXPECTED_MAX_NEW_TOKENS}")
    temperature = float(result["temperature"])
    top_p = float(result["top_p"])
    if temperature != 1.0 or top_p != 1.0:
        raise ValueError(f"{path}: decoding settings differ from frozen E2c")
    model_value = result.get("model")
    if not model_value:
        raise ValueError(f"{path}: missing evaluated model path")
    model_path = Path(model_value).resolve()
    reported_model_files = result.get("model_files", {})
    verified_model_files = {}
    for name in ("config.json", "model.safetensors"):
        model_file = model_path / name
        if not model_file.is_file():
            raise ValueError(
                f"{path}: evaluated checkpoint artifact is missing: "
                f"{model_file}")
        digest = _sha256_file(model_file)
        reported = reported_model_files.get(name, {})
        if (reported.get("sha256") != digest or
                int(reported.get("bytes", -1)) != model_file.stat().st_size):
            raise ValueError(
                f"{path}: evaluated checkpoint fingerprint drifted: {name}")
        verified_model_files[name] = {
            "bytes": model_file.stat().st_size,
            "sha256": digest,
        }
    if result.get("evaluator_sha256") != \
            FROZEN_ASSET_SHA256["eval_countdown.py"]:
        raise ValueError(f"{path}: evaluator implementation is not frozen E2c")
    if result.get("reward_sha256") != \
            FROZEN_ASSET_SHA256["countdown_reward.py"]:
        raise ValueError(f"{path}: reward implementation is not frozen E2c")
    manifest_bytes = json.dumps(
        task_manifest, separators=(",", ":")).encode("utf-8")
    provenance = {
        "result": str(path),
        "raw_outcomes": str(raw_path),
        "model": str(model_path),
        "model_files": verified_model_files,
        "evaluator_sha256": result["evaluator_sha256"],
        "reward_sha256": result["reward_sha256"],
        "data_sha256": data_sha256,
        "task_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "evaluation_seed": int(result["seed"]),
        "k": k,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "metric_definition": (
            "standard observed-set pass@16 from 16 retained binary outcomes; "
            "not VERL bootstrap best@16"
        ),
    }
    return recomputed["countdown_tier1"], provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", required=True)
    parser.add_argument("--readiness", required=True)
    parser.add_argument(
        "--result", action="append", required=True,
        help="seed,arm,path; provide B1, B2, and E2c for each seed")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.delivery).open(encoding="utf-8") as handle:
        delivery = json.load(handle)
    if (delivery.get("status") != "pass" or
            not delivery.get("endpoint_evaluation_permitted")):
        raise ValueError("E2c delivery did not pass; endpoints are forbidden")
    delivery_seeds = [int(item["seed"]) for item in delivery.get("seeds", [])]
    require_frozen_seeds(delivery_seeds)
    if any(item.get("status") != "pass" for item in delivery["seeds"]):
        raise ValueError("an E2c seed lacks passing treatment delivery")
    expected_displacement = summarize_displacement(delivery["seeds"])
    if delivery.get("displaced_slot_diagnostic") != expected_displacement:
        raise ValueError("E2c displaced-slot diagnostic is missing or drifted")
    readiness_path = Path(args.readiness).resolve()
    with readiness_path.open(encoding="utf-8") as handle:
        readiness = json.load(handle)
    if (readiness.get("audit_kind") !=
            "outcome_blind_e2c_launch_readiness" or
            readiness.get("integrity_status") != "pass" or
            readiness.get("delivery_gate", {}).get("status") != "pass" or
            readiness.get("heldout_artifacts_inspected") is not False):
        raise ValueError("no valid outcome-blind E2c readiness receipt")
    receipt_runs = {
        item["run_id"]: item
        for item in (readiness.get("comparators", []) +
                     readiness.get("e2c_runs", []))
    }

    endpoints = {}
    provenance = {}
    for specification in args.result:
        seed_text, arm, path_text = specification.split(",", 2)
        seed = int(seed_text)
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm}")
        key = (seed, arm)
        if key in endpoints:
            raise ValueError(f"duplicate result {key}")
        path = Path(path_text).resolve()
        endpoints[key], provenance[key] = load_endpoint_result(path)

    seeds = sorted({seed for seed, _ in endpoints})
    require_frozen_seeds(seeds)
    required = {(seed, arm) for seed in seeds for arm in ARMS}
    if set(endpoints) != required:
        missing = sorted(required - set(endpoints))
        extra = sorted(set(endpoints) - required)
        raise ValueError(
            f"need complete seeds {FROZEN_SEEDS}; missing={missing}, "
            f"extra={extra}")

    data_hashes = {item["data_sha256"] for item in provenance.values()}
    if len(data_hashes) != 1:
        raise ValueError("endpoint evaluations do not share one held-out dataset")
    task_hashes = {
        item["task_manifest_sha256"] for item in provenance.values()}
    if len(task_hashes) != 1:
        raise ValueError("endpoint evaluations do not share one task manifest")
    raw_paths = {item["raw_outcomes"] for item in provenance.values()}
    if len(raw_paths) != len(provenance):
        raise ValueError("endpoint arms reuse a raw-outcome artifact")
    for seed in seeds:
        paired = [provenance[(seed, arm)] for arm in ARMS]
        pairing_keys = (
            "evaluation_seed", "k", "temperature", "top_p", "max_new_tokens")
        for field in pairing_keys:
            if len({item[field] for item in paired}) != 1:
                raise ValueError(
                    f"seed {seed}: endpoint evaluations are not paired on {field}")
        if paired[0]["evaluation_seed"] != 10_000 + seed:
            raise ValueError(
                f"seed {seed}: evaluation seed is not frozen at {10_000 + seed}")
        for arm in ARMS:
            expected_run = (
                f"e2c_reservoir_replay_s{seed}_260810" if arm == "e2c"
                else f"e2_clean_{arm}_s{seed}_260809")
            if expected_run not in Path(provenance[(seed, arm)]["model"]).parts:
                raise ValueError(
                    f"seed {seed} {arm}: summary names the wrong checkpoint")
            receipt = receipt_runs.get(expected_run, {})
            if receipt.get("status") != "complete":
                raise ValueError(
                    f"seed {seed} {arm}: checkpoint lacks a complete receipt")
            expected_fingerprint = receipt.get(
                "checkpoint_fingerprint", {}).get("model.safetensors", {})
            if expected_fingerprint != provenance[(seed, arm)][
                    "model_files"]["model.safetensors"]:
                raise ValueError(
                    f"seed {seed} {arm}: endpoint model differs from receipt")

    seed_rows = []
    for seed in seeds:
        row = {
            "seed": seed,
            "endpoints": {arm: endpoints[(seed, arm)] for arm in ARMS},
            "contrasts": {},
        }
        for label, left, right in (
            ("b2_minus_b1", "b2", "b1"),
            ("e2c_minus_b1", "e2c", "b1"),
            ("e2c_minus_b2", "e2c", "b2"),
        ):
            row["contrasts"][label] = {
                metric: endpoints[(seed, left)][metric] -
                endpoints[(seed, right)][metric]
                for metric in METRICS
            }
        seed_rows.append(row)

    contrast_summary = {}
    for label in ("b2_minus_b1", "e2c_minus_b1", "e2c_minus_b2"):
        contrast_summary[label] = {
            metric: {
                "mean": mean(
                    row["contrasts"][label][metric] for row in seed_rows),
                "seed_values": [
                    row["contrasts"][label][metric] for row in seed_rows],
                "positive_seeds": sum(
                    row["contrasts"][label][metric] > 0
                    for row in seed_rows),
                "negative_seeds": sum(
                    row["contrasts"][label][metric] < 0
                    for row in seed_rows),
            } for metric in METRICS
        }

    direction_mean = contrast_summary["e2c_minus_b2"]["mean@16"]["mean"]
    direction_coverage = contrast_summary[
        "e2c_minus_b2"]["pass@16"]["mean"]
    report = {
        "status": "complete_descriptive_n3",
        "independent_unit": "training seed",
        "metric_provenance": (
            "Both endpoints were recomputed from retained per-task binary "
            "outcomes. pass@16 is standard observed-set coverage at n=k=16, "
            "not the historical VERL bootstrap best@16 proxy."
        ),
        "primary": "tier-1 e2c_minus_b2 mean@16",
        "safety": "tier-1 e2c_minus_b2 pass@16",
        "seed_results": seed_rows,
        "contrast_summary": contrast_summary,
        "direction_readout": {
            "mean@16": ("e2c_higher" if direction_mean > 0 else
                        "b2_higher" if direction_mean < 0 else "tie"),
            "pass@16": ("e2c_higher" if direction_coverage > 0 else
                        "b2_higher" if direction_coverage < 0 else "tie"),
            "note": (
                "This sign readout is not an equivalence test. Apply only the "
                "prospectively frozen manuscript branch supported by both "
                "endpoints and all seed-level values."),
        },
        "delivery": str(Path(args.delivery).resolve()),
        "outcome_blind_readiness": str(readiness_path),
        "delivery_interpretation": expected_displacement,
        "inputs": {f"seed_{seed}_{arm}": provenance[(seed, arm)]
                   for seed in seeds for arm in ARMS},
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
