#!/usr/bin/env python3
"""Fail-closed validation for the outcome-blind N-factorial DRAFT package.

This validator reads configuration and provenance metadata only.  It never
opens a checkpoint, training log, evaluation file, or performance endpoint.
The runtime lane must be invoked with the pinned CPU environment against a
fresh clone to which the exact v4 applicator has just been applied.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "ued_benchmark/analysis/development_protocol_v3_n_factorial_tie_aware_draft.json"
MANIFEST_PATH = ROOT / "ued_benchmark/analysis/n_factorial_tie_aware_v4_draft_manifest.json"
PROTOCOL_SHA256 = "81a57668d3cfdf595f13710df6152a437b8c4640791fbeeed2ef8c9e9486f26f"
MANIFEST_SHA256 = "58e1ffd9c7e3d80992971b331c540d6c8976c9cd4082391fae92de0df4fd417f"
CORE_CONTRACT_SHA256 = "3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b"
BASE_COMMIT = "d053054c5290a04c1c4cd8b55704d999cad73e30"
BASE_TREE = "b0cace1fc54984e21a842f12d15d0b899e33d270"
OVERLAY_VERSION = "frontier-activity-tie-aware-v4"
APPLIED_MANIFEST_SHA256 = "9b411f61ebc56bb93fc22cad6b19299c38eab2b696fa17f7783c7729e1db02ae"

CONFIG_HASHES = {
    "ued_benchmark/configs/maze_frontier_factorial_n2_16x2_b2000_tie_aware_v4.json":
        "2e443515d3876ad8c8a632d9cc21f2a92288adf971cde0b2c4751679eed32791",
    "ued_benchmark/configs/maze_maxmc_factorial_n2_16x2_b2000_tie_aware_v4.json":
        "81e2af766e588896c3013de23f22906446e27e18661e2dfc6b6f2cc4e284f1b3",
    "ued_benchmark/configs/maze_frontier_factorial_n4_8x4_b1000_tie_aware_v4.json":
        "181ca0210ad988a699d408827b15941ef0a6b9c1588f4abd8c12dc1e6cc706b5",
    "ued_benchmark/configs/maze_maxmc_factorial_n4_8x4_b1000_tie_aware_v4.json":
        "9033de1f79ee7f64ac980ebe28e90542e3cfe93dddf63374c20e1e52def824fb",
    "ued_benchmark/configs/maze_frontier_factorial_n8_4x8_b500_tie_aware_v4.json":
        "5cdaf48da9b6e3f2ab9dd0b9dd8c94eb7e49fe07d7744fc46b7b5f735b3a436d",
    "ued_benchmark/configs/maze_maxmc_factorial_n8_4x8_b500_tie_aware_v4.json":
        "105c6695baf86b894d65c6756fc5647d560c84daaef35ee3d3859c1eb9f68090",
}

# These bytes predate this package.  Their raw hashes make accidental edits a
# validator failure, including the source-faithful MaxMC reference and both
# prior development protocols.
PROTECTED_HASHES = {
    "ued_benchmark/UPSTREAM_PIN.json":
        "375ff36d64a98dd72f9b94f8bf7e63ae2cb6ec99571de37c7a8d483a936401d7",
    "ued_benchmark/OVERLAY_CONTRACT.json":
        "5868d346ba00e43225424f053ded3dea056b8f76ce4bfdaa1a107a4955cb9000",
    "ued_benchmark/OVERLAY_CONTRACT_V4.json": CORE_CONTRACT_SHA256,
    "ued_benchmark/OVERLAY_LINEAGE.json":
        "784e2fd1f545d49c8d10c3f3aeda37aae51fa00127e2c14578702e275bfb6971",
    "ued_benchmark/scripts/apply_minimax_overlay.py":
        "ddd3569b86adb703c8c7141fe7f2dae7a49c2c6b08e326edd61c3e3da7a345f7",
    "ued_benchmark/scripts/apply_minimax_overlay_v4.py":
        "c2e5eb3dac02b86723ece485cd348832f1636198c781bae82c1d99df0167590b",
    "ued_benchmark/overlay/minimax/util/rl/frontier_activity.py":
        "63726251813bd9fafc2722409c4a2942c6ae2728327870797df47d01504738ca",
    "ued_benchmark/overlay/minimax/util/rl/tie_aware_rank.py":
        "1b9db20d05edd3212346e84d14606af91ae443c0665945a7b679ade161560244",
    "ued_benchmark/analysis/development_protocol_v1.json":
        "9d0ccbeaf83564958c5374e6e68793aa644013b1e9f6b889a91da69c99a720ba",
    "ued_benchmark/analysis/development_protocol_v2_tie_aware_draft.json":
        "1e4bd62be2412fa5291fde9d2c8750f30ed2e9c9f43afcda93d8ab552e4a3269",
    "ued_benchmark/configs/maze_frontier_exact_grouped_n8.json":
        "b49168142a9d5a5d8edce88634975ac52d7615dca8bad9fff1cbcaf29ec43508",
    "ued_benchmark/configs/maze_maxmc_group_matched_4x8_b500.json":
        "6ec2083745ccc585383170f0a14f464397614a4365ba644e5c9e7e4ef422d943",
    "ued_benchmark/configs/maze_frontier_exact_grouped_n8_tie_aware_v4.json":
        "0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2",
    "ued_benchmark/configs/maze_maxmc_group_matched_4x8_b500_tie_aware_v4.json":
        "a3cc3ddf387a3bb7cf3d9759c3f13e6a74f2c7e32de311ac344d54ef5e703ec6",
    "ued_benchmark/configs/maze_maxmc_upstream_official_reference_32x1_b4000.json":
        "a5b8b87799bce31564959b3e8b55cfdaba658b31b8135fa35e3b96704d65185b",
    "ued_benchmark/configs/maze_maxmc_v4_stable_rank_compat_32x1_b4000.json":
        "99def895587a85e2ad060c356bf53041cf1fb6d1140304496451748cce207c92",
    "ued_benchmark/configs/maze_frontier_posterior_bridge_n8_neval1.json":
        "581369156855cf58718a686ea849df71410253b09561aaf83a68a6353151c883",
}

LAYOUTS = {
    2: {"n_eval": 2, "n_parallel": 16, "buffer_size": 2000},
    4: {"n_eval": 4, "n_parallel": 8, "buffer_size": 1000},
    8: {"n_eval": 8, "n_parallel": 4, "buffer_size": 500},
}
FRONTIER_ONLY_FIELDS = {
    "plr_frontier_n_rollouts",
    "plr_frontier_require_n_eval_match",
    "plr_frontier_prior_alpha",
    "plr_frontier_prior_beta",
    "plr_frontier_success_threshold",
    "plr_frontier_posterior_mode",
}
WITHIN_N_ALLOWED_DIFFERENCES = {"ued_score"} | FRONTIER_ONLY_FIELDS
ACROSS_N_ALLOWED_DIFFERENCES = {
    "n_eval", "n_parallel", "plr_buffer_size", "ued_score"
} | FRONTIER_ONLY_FIELDS
EXPECTED_VERSIONS = {
    "jax": "0.4.31",
    "jaxlib": "0.4.31",
    "flax": "0.8.5",
    "chex": "0.1.86",
    "optax": "0.2.3",
    "numpy": "1.25.2",
}


class ValidationError(RuntimeError):
    """Raised when the frozen engineering package fails closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"unsafe or missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"unsafe or missing {label}: {path}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid {label}: {path}") from exc
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def singleton_args(document: Mapping[str, Any], label: str) -> dict[str, Any]:
    require(set(document) == {"args"}, f"{label} top-level shape drift")
    grid = document["args"]
    require(isinstance(grid, dict) and grid, f"{label} args must be a nonempty object")
    result: dict[str, Any] = {}
    for key, values in grid.items():
        require(isinstance(key, str), f"{label} has a non-string field")
        require(isinstance(values, list) and len(values) == 1, f"{label} {key} is not singleton")
        result[key] = values[0]
    return result


def _without(values: Mapping[str, Any], fields: set[str]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if key not in fields}


def validate_static_package() -> dict[str, Any]:
    require(sha256(PROTOCOL_PATH) == PROTOCOL_SHA256, "factorial protocol hash drift")
    require(sha256(MANIFEST_PATH) == MANIFEST_SHA256, "factorial package manifest hash drift")
    for relative, expected in PROTECTED_HASHES.items():
        require(sha256(ROOT / relative) == expected, f"protected artifact hash drift: {relative}")

    protocol = load_json(PROTOCOL_PATH, "factorial protocol")
    manifest = load_json(MANIFEST_PATH, "factorial package manifest")
    require(protocol["schema"] == 1, "protocol schema drift")
    require(protocol["status"] == "DRAFT_ENGINEERING_ONLY_NOT_PRODUCTION_AUTHORIZED", "protocol status drift")
    require(protocol["production_driver_authorized"] is False, "production authorization drift")
    require(protocol["endpoint_access_authorized"] is False, "endpoint authorization drift")
    require(protocol["paper_evidence"] is False, "paper-evidence status drift")
    provenance = protocol["provenance"]
    require(provenance["base_commit"] == BASE_COMMIT, "protocol base commit drift")
    require(provenance["base_tree"] == BASE_TREE, "protocol base tree drift")
    require(provenance["overlay_version"] == OVERLAY_VERSION, "protocol overlay version drift")
    require(provenance["core_contract"]["sha256"] == CORE_CONTRACT_SHA256, "core dependency drift")
    require(
        provenance["expected_fresh_applied_overlay_manifest_sha256"] == APPLIED_MANIFEST_SHA256,
        "applied-overlay manifest dependency drift",
    )

    require(manifest["protocol"] == {
        "path": str(PROTOCOL_PATH.relative_to(ROOT)), "sha256": PROTOCOL_SHA256
    }, "manifest protocol closure drift")
    require(manifest["core_dependency"]["sha256"] == CORE_CONTRACT_SHA256, "manifest core drift")
    require(manifest["production_authorized"] is False, "manifest production authorization drift")
    require(manifest["endpoint_access_authorized"] is False, "manifest endpoint authorization drift")
    manifest_configs = {item["path"]: item["sha256"] for item in manifest["configs"]}
    require(manifest_configs == CONFIG_HASHES, "manifest config closure drift")
    manifest_protected = {
        item["path"]: item["sha256"] for item in manifest["protected_prior_artifacts"]
    }
    expected_manifest_protected = {
        relative: PROTECTED_HASHES[relative]
        for relative in (
            "ued_benchmark/analysis/development_protocol_v1.json",
            "ued_benchmark/analysis/development_protocol_v2_tie_aware_draft.json",
            "ued_benchmark/configs/maze_maxmc_upstream_official_reference_32x1_b4000.json",
            "ued_benchmark/configs/maze_frontier_posterior_bridge_n8_neval1.json",
        )
    }
    require(manifest_protected == expected_manifest_protected, "manifest protected-artifact closure drift")

    cells = protocol["cells"]
    require(isinstance(cells, list) and len(cells) == 6, "factorial must contain six cells")
    by_N: dict[int, dict[str, tuple[dict[str, Any], Mapping[str, Any]]]] = {}
    all_normalized: list[dict[str, Any]] = []
    xpids: set[str] = set()
    seen_paths: set[str] = set()
    for cell in cells:
        require(isinstance(cell, dict), "cell must be an object")
        N = cell["N"]
        arm = cell["arm"]
        require(N in LAYOUTS and arm in {"frontier", "maxmc"}, "unknown factorial cell")
        relative = cell["config_path"]
        require(relative in CONFIG_HASHES and relative not in seen_paths, "cell config path drift")
        seen_paths.add(relative)
        require(cell["config_sha256"] == CONFIG_HASHES[relative], "protocol config hash drift")
        require(sha256(ROOT / relative) == CONFIG_HASHES[relative], f"config hash drift: {relative}")
        document = load_json(ROOT / relative, f"{cell['cell_id']} config")
        authored = singleton_args(document, f"{cell['cell_id']} config")
        layout = LAYOUTS[N]
        require(authored["n_eval"] == layout["n_eval"], f"{cell['cell_id']} n_eval drift")
        require(authored["n_parallel"] == layout["n_parallel"], f"{cell['cell_id']} n_parallel drift")
        require(authored["plr_buffer_size"] == layout["buffer_size"], f"{cell['cell_id']} buffer drift")
        require(authored["n_eval"] * authored["n_parallel"] == 32, f"{cell['cell_id']} stream drift")
        require(authored["n_rollout_steps"] == 256, f"{cell['cell_id']} rollout drift")
        require(authored["plr_buffer_size"] * authored["n_eval"] == 4000, f"{cell['cell_id']} capacity scaling drift")
        require(authored["plr_min_fill_ratio"] == 0.5, f"{cell['cell_id']} min-fill drift")
        require(authored["plr_tie_aware_score_ranks"] is True, f"{cell['cell_id']} tie-aware rank drift")
        require(authored["plr_use_score_ranks"] is True, f"{cell['cell_id']} score-rank drift")
        require(authored["plr_temp"] == 0.3, f"{cell['cell_id']} temperature drift")
        require(authored["plr_staleness_coef"] == 0.3, f"{cell['cell_id']} staleness drift")
        require(authored["plr_replay_prob"] == 0.5, f"{cell['cell_id']} replay-probability drift")
        require(authored["from_last_checkpoint"] is False, f"{cell['cell_id']} resume drift")
        require(
            authored["plr_frontier_overlay_contract_sha256"] == CORE_CONTRACT_SHA256,
            f"{cell['cell_id']} core binding drift",
        )
        require(authored["plr_frontier_overlay_version"] == OVERLAY_VERSION, f"{cell['cell_id']} overlay drift")
        if arm == "frontier":
            require(authored["ued_score"] == "coefficient_activity", "Frontier score drift")
            require(authored["plr_frontier_n_rollouts"] == authored["n_eval"] == N, "strict Frontier N drift")
            require(authored["plr_frontier_require_n_eval_match"] is True, "strict Frontier mode drift")
            require(authored["plr_frontier_posterior_mode"] == "expected_activity", "posterior drift")
            require(authored["plr_frontier_prior_alpha"] == 1.0, "Frontier alpha drift")
            require(authored["plr_frontier_prior_beta"] == 1.0, "Frontier beta drift")
            require(authored["plr_frontier_success_threshold"] == 0.0, "success threshold drift")
        else:
            require(authored["ued_score"] == "max_mc", "MaxMC score drift")
            require(not (FRONTIER_ONLY_FIELDS & set(authored)), "MaxMC authors hidden Frontier fields")
        by_N.setdefault(N, {})[arm] = (authored, cell)
        all_normalized.append(_without(authored, ACROSS_N_ALLOWED_DIFFERENCES))
        xpid = cell["grid_generated_xpid"]
        require(isinstance(xpid, str) and xpid and xpid not in xpids, "xpid identity drift")
        xpids.add(xpid)

    require(seen_paths == set(CONFIG_HASHES), "protocol/config closure is incomplete")
    for N, arms in by_N.items():
        require(set(arms) == {"frontier", "maxmc"}, f"N={N} pair is incomplete")
        frontier = arms["frontier"][0]
        maxmc = arms["maxmc"][0]
        require(
            _without(frontier, WITHIN_N_ALLOWED_DIFFERENCES)
            == _without(maxmc, WITHIN_N_ALLOWED_DIFFERENCES),
            f"N={N} arms differ outside score/Frontier-only fields",
        )
        differing = {
            key for key in set(frontier) | set(maxmc)
            if frontier.get(key, object()) != maxmc.get(key, object())
        }
        require(differing == WITHIN_N_ALLOWED_DIFFERENCES, f"N={N} authored difference set drift")
    reference = all_normalized[0]
    require(all(item == reference for item in all_normalized[1:]), "common factorial knobs drift across N")

    warm = protocol["nominal_warm_fill"]
    require(
        warm["classification"] == "exact_only_conditioned_on_distinct_accepted_new_groups",
        "warm-fill condition is not fail-closed",
    )
    for layout in LAYOUTS.values():
        threshold = math.ceil(layout["buffer_size"] * 0.5)
        cycles = math.ceil(threshold / layout["n_parallel"])
        require(cycles == 63, "nominal warm-fill cycle math drift")
        require(cycles * 32 * 256 == 516096, "nominal warm-fill transition math drift")
    gate = warm["receipt_gate"]
    require("zero duplicate-new groups" in gate["exact_63_cycle_label_requires"], "duplicate gate missing")
    require("observed" in gate["if_gate_fails"], "observed-fill fallback missing")

    reporting = protocol["reporting"]
    cycles = reporting["fixed_transition_secondary"]["outer_cycle_grid"]
    transitions = reporting["fixed_transition_secondary"]["exact_training_transition_grid"]
    require(transitions == [cycle * 8192 for cycle in cycles], "fixed-transition grid drift")
    require(reporting["fixed_update_primary"]["target_student_ppo_updates"] == 30000, "update target drift")
    require(reporting["fixed_update_primary"]["target_upstream_n_grad_updates"] == 30000, "upstream n_grad_updates target drift")
    pair_gate = reporting["fixed_update_primary"]["within_N_seed_pair_validity_gate"]
    require(
        pair_gate["must_equal_exactly"] == [
            "terminal n_updates = 30000",
            "terminal upstream n_grad_updates = 30000",
            "terminal optimizer step applications = 150000",
            "outer-cycle count",
            "training-transition count",
            "the complete set of aligned fixed-transition cycle observations",
        ],
        "within-N matched-budget gate drift",
    )
    require("invalidate" in pair_gate["failure_policy"], "matched-budget failure policy drift")
    require(
        reporting["fixed_update_primary"]["across_N_terminal_cycle_or_transition_equality_required"] is False,
        "cross-N terminal equality interpretation drift",
    )
    require("all six cells" in reporting["fixed_transition_secondary"]["across_N_alignment"], "cross-N transition alignment drift")
    identity = protocol["matched_training_contract"]["execution_identity_and_path_contract"]
    require(identity["direct_parser_default_xpid"] == "latest", "parser-default xpid drift")
    require(identity["direct_parser_launch_is_safe"] is False, "direct launch was incorrectly authorized")
    require(identity["path_overlap_or_xpid_latest"] == "invalidate before training", "identity failure gate drift")
    terminal = protocol["terminal_checkpoint_contract"]
    require(terminal["resume_allowed"] is False, "resume authorization drift")
    require(terminal["authored_from_last_checkpoint"] is False, "authored resume drift")
    require(terminal["terminal_checkpoint_required"] is True, "terminal checkpoint requirement drift")
    require(terminal["periodic_checkpoint_is_admissible_endpoint"] is False, "periodic endpoint substitution drift")
    require(terminal["round_trip_required_before_evaluation"] is True, "checkpoint round-trip gate drift")
    round_trip = " ".join(terminal["round_trip_must_bind_exactly"])
    for token in (
        "manifest", "protocol", "config", "arm", "N", "n_eval",
        "n_updates", "n_grad_updates", "PLR buffer",
    ):
        require(token in round_trip, f"checkpoint round-trip binding missing: {token}")
    require("cannot be used for recovery or resume" in terminal["periodic_checkpoint_role"], "periodic recovery prohibition drift")
    prerequisites = " ".join(protocol["prerequisites_for_any_future_run"])
    for component in (
        "trainer", "evaluator", "assembler or finalizer", "analyzer",
        "scheduler script", "source bundle", "complete environment provenance",
    ):
        require(component in prerequisites, f"future campaign binding missing: {component}")
    require(protocol["development_design"]["paired_training_seeds"] == [101, 102, 103, 104, 105], "development seed drift")
    require(protocol["development_design"]["descriptive_only"] is True, "development inference drift")
    require(len(protocol["estimands"]["primary_within_N"]) == 3, "within-N contrast family drift")
    require(len(protocol["estimands"]["predeclared_contrast_of_contrasts"]) == 2, "interaction contrast drift")
    require("cannot identify an estimator-N-only effect" in protocol["estimands"]["identifiability"], "N identifiability limit drift")
    require(protocol["multiplicity_and_confirmatory_hold"]["current_package_authorizes_none_of_these"] is True, "confirmatory hold drift")
    return {"protocol": protocol, "configs": by_N}


def _git(source: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=source, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _git_raw(source: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=source, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout


def validate_fresh_source(source: Path) -> dict[str, Any]:
    source = source.resolve()
    require(source.is_dir() and not source.is_symlink(), "unsafe or missing applied source")
    require((source / ".git").exists(), "fresh-source validation requires a Git clone")
    require(_git(source, "rev-parse", "HEAD") == BASE_COMMIT, "applied source commit drift")
    require(_git(source, "rev-parse", "HEAD^{tree}") == BASE_TREE, "applied source base tree drift")
    applied_path = source / ".frontierrl_overlay.json"
    require(sha256(applied_path) == APPLIED_MANIFEST_SHA256, "applied manifest hash drift")
    applied = load_json(applied_path, "applied overlay manifest")
    require(applied["base_commit"] == BASE_COMMIT, "applied manifest commit drift")
    require(applied["overlay"] == OVERLAY_VERSION, "applied manifest version drift")
    require(applied["overlay_contract_sha256"] == CORE_CONTRACT_SHA256, "applied manifest core drift")
    overlay_files = applied["overlay_files"]
    require(isinstance(overlay_files, list) and len(overlay_files) == len(set(overlay_files)), "overlay file closure drift")
    require(set(applied["overlay_file_sha256"]) == set(overlay_files), "overlay hash closure drift")
    for relative in overlay_files:
        require(sha256(source / relative) == applied["overlay_file_sha256"][relative], f"applied file drift: {relative}")
    status_paths = set()
    for line in _git_raw(source, "status", "--porcelain=v1", "--untracked-files=all").splitlines():
        require(len(line) >= 4, "malformed fresh-source Git status")
        status_paths.add(line[3:])
    require(status_paths == set(overlay_files) | {".frontierrl_overlay.json"}, "fresh source has unrelated changes")
    return applied


def _encoded_cli(value: Any) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    return str(value)


def validate_pinned_cpu_runtime(source: Path, static: Mapping[str, Any]) -> dict[str, Any]:
    require(sys.version_info[:2] == (3, 10), "pinned runtime requires Python 3.10")
    require(sys.flags.optimize == 0, "optimized Python is forbidden")
    require(sys.flags.no_user_site == 1, "user site must be disabled")
    require(sys.dont_write_bytecode, "bytecode writes must be disabled")
    for package, expected in EXPECTED_VERSIONS.items():
        require(importlib.metadata.version(package) == expected, f"pinned package drift: {package}")
    require(os.environ.get("JAX_PLATFORMS") == "cpu", "JAX_PLATFORMS must be exactly cpu")
    require("minimax" not in sys.modules, "minimax was imported before source validation")
    sys.path.insert(0, str(source.resolve() / "src"))

    import jax
    import jax.numpy as jnp
    from minimax.arguments import parser
    from minimax.config.make_cmd import (
        generate_all_params_for_grid,
        generate_train_cmds,
        xpid_from_params,
    )
    from minimax.util.rl import PLRManager, UEDScore, VmapTrainState

    require(jax.__version__ == "0.4.31", "imported JAX version drift")
    require(jax.default_backend() == "cpu", "non-CPU JAX backend is forbidden")
    require(all(device.platform == "cpu" for device in jax.devices()), "non-CPU JAX device found")

    parsed_xpids: set[str] = set()
    for N in sorted(static["configs"]):
        for arm in ("frontier", "maxmc"):
            authored, cell = static["configs"][N][arm]
            previous = sys.argv
            try:
                sys.argv = ["validate-n-factorial"] + [
                    f"--{key}={_encoded_cli(value)}" for key, value in authored.items()
                ]
                resolved = parser.parse_args()
            finally:
                sys.argv = previous
            runner = resolved.train_runner_args
            require(resolved.xpid == "latest", f"parser-default xpid drift: {cell['cell_id']}")
            require(runner.n_eval == N, f"parsed n_eval drift: {cell['cell_id']}")
            require(runner.n_parallel == LAYOUTS[N]["n_parallel"], f"parsed n_parallel drift: {cell['cell_id']}")
            require(runner.buffer_size == LAYOUTS[N]["buffer_size"], f"parsed buffer drift: {cell['cell_id']}")
            require(runner.tie_aware_score_ranks is True, f"parsed tie-aware drift: {cell['cell_id']}")
            require(runner.frontier_overlay_contract_sha256 == CORE_CONTRACT_SHA256, "parsed core drift")
            if arm == "frontier":
                require(runner.ued_score == "coefficient_activity", "parsed Frontier score drift")
                require(runner.frontier_n_rollouts == runner.n_eval == N, "parsed strict N mismatch")
                require(runner.frontier_require_n_eval_match is True, "parsed strict mode drift")
            else:
                require(runner.ued_score == "max_mc", "parsed MaxMC score drift")

            grid = {key: [value] for key, value in authored.items()}
            params = generate_all_params_for_grid(copy.deepcopy(grid))[0]
            command = generate_train_cmds(
                "train", params, num_trials=1, start_index=0,
                xpid_generator=xpid_from_params,
            )[0]
            match = re.search(r"(?:^| )--xpid=([^ ]+)", command)
            require(match is not None, f"generated xpid missing: {cell['cell_id']}")
            xpid = match.group(1)
            require(xpid == cell["grid_generated_xpid"], f"grid-generated xpid drift: {cell['cell_id']}")
            require(xpid not in parsed_xpids, "generated xpid collision")
            parsed_xpids.add(xpid)

    def frontier_buffer(n_rollouts: int, n_eval: int):
        manager = PLRManager(
            example_level={"probe": jnp.zeros((2,), dtype=jnp.int32)},
            ued_score=UEDScore.RETURN,
            replay_prob=0.5,
            buffer_size=8,
            staleness_coef=0.3,
            temp=0.3,
            min_fill_ratio=0.5,
            use_score_ranks=True,
            use_robust_plr=True,
            use_parallel_eval=False,
            tie_aware_score_ranks=True,
            use_frontier_activity=True,
            frontier_n_rollouts=n_rollouts,
            frontier_n_eval=n_eval,
            frontier_require_n_eval_match=True,
            frontier_prior_alpha=1.0,
            frontier_prior_beta=1.0,
            frontier_success_threshold=0.0,
            frontier_posterior_mode="expected_activity",
            frontier_overlay_version=OVERLAY_VERSION,
            frontier_overlay_contract_sha256=CORE_CONTRACT_SHA256,
            n_devices=1,
        )
        return manager.reset()

    def train_state(buffer: Any):
        zeros = jnp.zeros((1,), dtype=jnp.uint32)
        return VmapTrainState(
            n_iters=zeros,
            n_updates=zeros,
            n_grad_updates=zeros,
            apply_fn=lambda *_args, **_kwargs: None,
            params={},
            tx=None,
            opt_state=(),
            plr_buffer=buffer,
        )

    saved_n2 = train_state(frontier_buffer(2, 2)).state_dict
    current_n4 = train_state(frontier_buffer(4, 4))
    try:
        current_n4.load_state_dict(saved_n2)
    except ValueError as exc:
        require(
            str(exc) == "PLR checkpoint configuration mismatch: frontier_n_rollouts.",
            "checkpoint N mismatch rejected for the wrong reason",
        )
    else:
        raise ValidationError("checkpoint static N mismatch was accepted")

    try:
        frontier_buffer(2, 4)
    except ValueError as exc:
        require("frontier_n_eval must equal frontier_n_rollouts" in str(exc), "strict N mismatch reason drift")
    else:
        raise ValidationError("strict Frontier N/n_eval mismatch was accepted")

    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "packages": EXPECTED_VERSIONS,
        "jax_backend": jax.default_backend(),
        "jax_devices": [device.device_kind for device in jax.devices()],
        "parsed_config_count": len(parsed_xpids),
        "unique_xpid_count": len(parsed_xpids),
        "direct_parser_default_xpid": "latest",
        "direct_parser_launch_safe": False,
        "checkpoint_static_N_mismatch_rejected": True,
        "strict_frontier_N_n_eval_mismatch_rejected": True,
    }


def validate(source: Path) -> dict[str, Any]:
    static = validate_static_package()
    applied = validate_fresh_source(source)
    runtime = validate_pinned_cpu_runtime(source, static)
    return {
        "schema": 1,
        "status": "passed",
        "purpose": "outcome-blind N-factorial engineering package validation",
        "paper_evidence": False,
        "endpoint_accessed": False,
        "production_authorized": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "package_manifest_sha256": MANIFEST_SHA256,
        "core_contract_sha256": CORE_CONTRACT_SHA256,
        "config_sha256": dict(sorted(CONFIG_HASHES.items())),
        "protected_artifact_count": len(PROTECTED_HASHES),
        "fresh_applied_overlay_manifest_sha256": sha256(source.resolve() / ".frontierrl_overlay.json"),
        "fresh_applied_overlay_file_count": len(applied["overlay_files"]),
        "runtime": runtime,
    }


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="fresh pinned clone with exact v4 overlay applied")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        receipt = validate(parse_cli(argv).source_dir)
    except (ValidationError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"N_FACTORIAL_DRAFT_REFUSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
