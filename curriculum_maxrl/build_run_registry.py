"""Build and validate the paper's one-row-per-training-run registry.

The registry is intentionally generated from versioned, repository-relative
evidence.  A run may be represented by a raw vendored artifact or by a cell in
a vendored aggregate result.  Missing raw artifacts are never represented by
machine-local paths; their availability is explicit in ``raw_status``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = Path(__file__).with_name("run_registry.json")
GENERATED_DATE = "2026-08-09"


# The historical matched-clock cohort frozen for the paper.  These entries
# preserve the exact arm/seed accounting of the original registry.  When a raw
# file is not vendored, un_form_verdicts.json is the versioned result evidence.
MAZE_COHORT: tuple[tuple[str, str, int], ...] = (
    ("matched_falp_maxrl_hs_s0.jsonl", "falp_maxrl_hs", 0),
    ("matched_falp_maxrl_hsdense_s0.jsonl", "falp_maxrl_hsdense", 0),
    ("matched_falp_maxrl_hsdense_s1.jsonl", "falp_maxrl_hsdense", 1),
    ("matched_falp_maxrl_hsdense_s2.jsonl", "falp_maxrl_hsdense", 2),
    ("matched_falp_maxrl_hsdense_tt_s0.jsonl", "falp_maxrl_hsdense_tt", 0),
    ("matched_falp_p4_hsdense_s0.jsonl", "falp_p4_hsdense", 0),
    ("matched_frontier_alp_maxrl_s0.jsonl", "frontier_alp_maxrl", 0),
    ("matched_frontier_alp_maxrl_s1.jsonl", "frontier_alp_maxrl", 1),
    ("matched_frontier_alp_maxrl_s2.jsonl", "frontier_alp_maxrl", 2),
    ("matched_frontier_grpo_s0.jsonl", "frontier_grpo", 0),
    ("matched_frontier_maxrl_hs_s0.jsonl", "frontier_maxrl_hs", 0),
    ("matched_frontier_maxrl_hs_s1.jsonl", "frontier_maxrl_hs", 1),
    ("matched_frontier_maxrl_hs_s2.jsonl", "frontier_maxrl_hs", 2),
    ("matched_frontier_maxrl_s0.jsonl", "frontier_maxrl", 0),
    ("matched_frontier_maxrl_s1.jsonl", "frontier_maxrl", 1),
    ("matched_frontier_maxrl_s2.jsonl", "frontier_maxrl", 2),
    ("matched_frontier_un_maxrl_s0.jsonl", "frontier_un_maxrl", 0),
    ("matched_frontier_un_maxrl_s1.jsonl", "frontier_un_maxrl", 1),
    ("matched_frontier_un_maxrl_s2.jsonl", "frontier_un_maxrl", 2),
    ("matched_frontier_un_tilt_maxrl_s0.jsonl", "frontier_un_tilt_maxrl", 0),
    ("matched_frontier_un_tilt_maxrl_s1.jsonl", "frontier_un_tilt_maxrl", 1),
    ("matched_frontier_un_tilt_maxrl_s2.jsonl", "frontier_un_tilt_maxrl", 2),
    ("matched_learnability_maxrl_s0.jsonl", "learnability_maxrl", 0),
    ("matched_uniform_grpo_s0.jsonl", "uniform_grpo", 0),
    ("matched_uniform_grpo_s1.jsonl", "uniform_grpo", 1),
    ("matched_uniform_grpo_s2.jsonl", "uniform_grpo", 2),
    ("matched_uniform_maxrl_hs_s0.jsonl", "uniform_maxrl_hs", 0),
    ("matched_uniform_maxrl_s0.jsonl", "uniform_maxrl", 0),
    ("matched_uniform_maxrl_s1.jsonl", "uniform_maxrl", 1),
    ("matched_uniform_maxrl_s2.jsonl", "uniform_maxrl", 2),
    ("matched_wide_falp_hsdense_s0.jsonl", "wide_falp_hsdense", 0),
)

MAZE_CHECKPOINT_RUNS: tuple[tuple[str, str], ...] = (
    ("ck_frontier_alp_maxrl_hsd.jsonl", "frontier_alp_maxrl_hsd"),
    ("ck_uniform_grpo.jsonl", "uniform_grpo"),
    ("ck_uniform_maxrl.jsonl", "uniform_maxrl"),
)

COUNTDOWN_MAIN_ARMS = {
    "B1": "no_recycling",
    "B2": "ungated_recycling",
    "B3": "under_gated_recycling_buggy_decay",
}

COUNTDOWN_V1_CELLS = {
    "C1": "teacher_plus_recycling",
    "C2": "teacher_no_recycling",
    "C3": "uniform_no_recycling",
    "C4": "uniform_plus_recycling",
}

ACROBOT_TOURNAMENT_LOCK = "frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json"
ACROBOT_TOURNAMENT_DEVELOPMENT = (
    "frontier_rl/examples/acrobot_curriculum_tournament_development.json"
)
ACROBOT_TOURNAMENT_GATE = (
    "frontier_rl/examples/acrobot_curriculum_tournament_development_gates.json"
)
ACROBOT_TOURNAMENT_CONFIRMATORY = (
    "frontier_rl/examples/acrobot_curriculum_tournament_confirmatory.json"
)
ACROBOT_TOURNAMENT_ANALYSIS = (
    "frontier_rl/examples/acrobot_curriculum_tournament_analysis.json"
)
ACROBOT_TOURNAMENT_ARMS: tuple[tuple[str, str, str], ...] = (
    ("uniform_shared_h64", "uniform", "uniform"),
    ("p1mp_shared_h64", "p1mp", "p1mp"),
    ("u16_shared_h64", "u16", "u16"),
)

ACROBOT_PROCURL_LOCK = "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_LOCK.json"
ACROBOT_PROCURL_DEVELOPMENT = (
    "frontier_rl/examples/acrobot_procurl_selection_development.json"
)
ACROBOT_PROCURL_GATE = (
    "frontier_rl/examples/acrobot_procurl_selection_development_gates.json"
)
ACROBOT_PROCURL_ANALYSIS = (
    "frontier_rl/examples/acrobot_procurl_selection_analysis.json"
)
ACROBOT_PROCURL_PORTABLE = (
    "frontier_rl/examples/acrobot_procurl_selection_portable_verification.json"
)
ACROBOT_PROCURL_DIAGNOSTICS = (
    "frontier_rl/examples/acrobot_procurl_selection_diagnostics.json"
)
ACROBOT_PROCURL_EXTERNAL_MANIFEST = (
    "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_EXTERNAL_RAW_MANIFEST.json"
)
ACROBOT_PROCURL_RESULTS = (
    "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_RESULTS.md"
)
ACROBOT_PROCURL_EXTERNAL_RAW = (
    "frontier_rl/examples/acrobot_procurl_selection_confirmatory.json"
)
ACROBOT_PROCURL_ARMS: tuple[tuple[str, str, bool], ...] = (
    ("procurl_env_b20_f5120", "procurl_p1mp_softmax", True),
    ("probe_sham_uniform_f5120", "uniform_sham", True),
    ("ordinary_uniform", "uniform_ordinary", False),
    ("u16_probe_range_matched_f5120", "u16_softmax", True),
)
ACROBOT_PROCURL_EXPECTED_SHA256 = {
    ACROBOT_PROCURL_LOCK: (
        "b7c7f76f6aaffa1fe65557717bfe545f2ec850495d370cd97fe72a8871fc8d0f"
    ),
    ACROBOT_PROCURL_DEVELOPMENT: (
        "6d9fa639295e35cd8a8da810ace82d330c863edace702db4e3f7d25a9ad82ba8"
    ),
    ACROBOT_PROCURL_GATE: (
        "1edf50dc0b86744b8e33a87afaee8f50dc05dd3ef0d3129bfb5eef2533cf34bf"
    ),
    ACROBOT_PROCURL_ANALYSIS: (
        "2010e30b5b15a212e2d6bdfaacd43d2434e5f468a96be613cf59744b9bc2fb38"
    ),
    ACROBOT_PROCURL_PORTABLE: (
        "c6b754655cbe6fa0dc52e065cd46d840f6a62b9bc58104ceb7443c727b0d01ae"
    ),
    ACROBOT_PROCURL_DIAGNOSTICS: (
        "583d950b8e85e6ea3efb477e6a390ceeac339054e46bb5ac69ff4597438d48c7"
    ),
    ACROBOT_PROCURL_EXTERNAL_MANIFEST: (
        "e197c1d581bcda8679ba4c0dc428fde10db6e80b653f1bb9e2dc02314e4dfdc6"
    ),
    ACROBOT_PROCURL_RESULTS: (
        "0063e929ba762679e35589dfc175786081f345aef10ea9ef7695f007b9c9c272"
    ),
}
ACROBOT_PROCURL_EXTERNAL_RAW_SHA256 = (
    "b1f8756c249effab8c77101c8bca73ddf708a5e143c18fe8742fd5712fdd7c12"
)
ACROBOT_PROCURL_EXTERNAL_RAW_SIZE = 1_374_886_097


def _repo_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl_rows(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{_repo_path(path)}:{line_number}: {exc}") from exc
            count += 1
    return count


def _row(
    *,
    run_id: str,
    suite: str,
    experiment: str,
    arm: str,
    seed: int | None,
    protocol: str,
    evidence_path: str,
    evidence_locator: str,
    raw_path: str | None,
    raw_status: str,
    status: str = "complete",
    n_eval_rows: int | None = None,
    verdict: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "run_id": run_id,
        "suite": suite,
        "experiment": experiment,
        "arm": arm,
        "seed": seed,
        "status": status,
        "protocol": protocol,
        "evidence_path": evidence_path,
        "evidence_locator": evidence_locator,
        "raw_path": raw_path,
        "raw_status": raw_status,
    }
    if n_eval_rows is not None:
        result["n_eval_rows"] = n_eval_rows
    if verdict is not None:
        result["verdict"] = verdict
    return result


def _maze_cohort_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    maze_dir = ROOT / "curriculum_maxrl" / "maze_gpu"
    un_evidence = "curriculum_maxrl/un_form_verdicts.json"

    for filename, arm, seed in MAZE_COHORT:
        raw = maze_dir / filename
        if raw.exists():
            evidence_path = raw_path = _repo_path(raw)
            evidence_locator = "complete JSONL artifact"
            raw_status = "vendored"
            n_eval_rows = _jsonl_rows(raw)
        else:
            if arm == "frontier_maxrl":
                summary_arm = "frontier_legacy"
            elif arm == "frontier_un_maxrl":
                summary_arm = "frontier_un"
            elif arm == "frontier_un_tilt_maxrl":
                summary_arm = "frontier_un_tilt"
            else:
                raise ValueError(f"No vendored evidence mapping for {filename}")
            evidence_path = un_evidence
            evidence_locator = f"arms.{summary_arm}.per_seed[seed={seed}]"
            raw_path = None
            raw_status = "external-not-vendored"
            n_eval_rows = None

        rows.append(
            _row(
                run_id=f"maze-cohort-{_slug(Path(filename).stem)}",
                suite="maze",
                experiment="matched_wall_clock_cohort",
                arm=arm,
                seed=seed,
                protocol="matched wall-clock 2400s",
                evidence_path=evidence_path,
                evidence_locator=evidence_locator,
                raw_path=raw_path,
                raw_status=raw_status,
                n_eval_rows=n_eval_rows,
            )
        )

    for filename, arm in MAZE_CHECKPOINT_RUNS:
        raw = maze_dir / filename
        rows.append(
            _row(
                run_id=f"maze-checkpoint-{_slug(Path(filename).stem)}",
                suite="maze",
                experiment="checkpoint_extension",
                arm=arm,
                seed=None,
                protocol="checkpoint extension; seed not recorded in vendored artifact",
                evidence_path=_repo_path(raw),
                evidence_locator="complete JSONL artifact",
                raw_path=_repo_path(raw),
                raw_status="vendored",
                n_eval_rows=_jsonl_rows(raw),
            )
        )
    return rows


def _factorial_sort_key(cell_key: str) -> tuple[int, str, str]:
    teacher, estimator, seed_token = cell_key.split("/")
    return int(seed_token.removeprefix("s")), teacher, estimator


def _maze_factorial_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for wave, filename in (
        (1, "results_factorial_wave1.json"),
        (2, "results_factorial_wave2.json"),
    ):
        path = ROOT / "curriculum_maxrl" / "maze_gpu_factorial" / filename
        cells = _json(path)["cells"]
        expected_count = 36 if wave == 1 else 24
        if len(cells) != expected_count:
            raise ValueError(
                f"{_repo_path(path)} has {len(cells)} cells, expected {expected_count}"
            )
        for cell_key in sorted(cells, key=_factorial_sort_key):
            teacher, estimator, seed_token = cell_key.split("/")
            seed = int(seed_token.removeprefix("s"))
            rows.append(
                _row(
                    run_id=(
                        f"maze-factorial-w{wave}-{_slug(teacher)}-"
                        f"{_slug(estimator)}-s{seed}"
                    ),
                    suite="maze",
                    experiment=f"balanced_factorial_wave_{wave}",
                    arm=f"{teacher}/{estimator}",
                    seed=seed,
                    protocol="250 fixed steps; eval every 25; lr=1e-4",
                    evidence_path=_repo_path(path),
                    evidence_locator=f"cells[{cell_key}]",
                    raw_path=None,
                    raw_status="external-at-execution-fork-9f7dd2e",
                )
            )
    return rows


def _countdown_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # The v1 predecessor is paper-used as a shallow-pool boundary result.  The
    # local analysis records four completed cells but does not record a seed,
    # so seed remains null rather than being inferred.
    for cell, arm in COUNTDOWN_V1_CELLS.items():
        rows.append(
            _row(
                run_id=f"countdown-v1-{cell.lower()}-{_slug(arm)}",
                suite="countdown",
                experiment="countdown_v1_2x2",
                arm=arm,
                seed=None,
                protocol="60 steps from shared SFT warmstart; seed not recorded locally",
                evidence_path="COUNTDOWN_ANALYSIS.md",
                evidence_locator=f"E-LLM-2 step-60 validation row {cell}",
                raw_path=None,
                raw_status="external-not-vendored",
            )
        )

    # Countdown-v2 main arms are explicitly three arms x seeds 1--3 in the
    # analysis.  Detailed raw logs are external; the versioned aggregate is
    # sufficient to establish completion and paper-used endpoints.
    for label, arm in COUNTDOWN_MAIN_ARMS.items():
        for seed in (1, 2, 3):
            main_row = _row(
                run_id=f"countdown-v2-{label.lower()}-{_slug(arm)}-s{seed}",
                suite="countdown",
                experiment="countdown_v2_sharpening",
                arm=arm,
                seed=seed,
                protocol="60 steps; step-60 evaluation; 16 samples/prompt",
                evidence_path="COUNTDOWN_ANALYSIS.md",
                evidence_locator=(f"3-seed aggregate: {label}, completed seeds 1--3"),
                raw_path=None,
                raw_status="external-not-vendored",
            )
            main_row["result_path"] = "paper/figures/data/b_scoreboard_3seed.json"
            main_row["result_locator"] = f"{label}_t1 and {label}_t2"
            main_row["metric_provenance_path"] = (
                "curriculum_maxrl/countdown_reviewer_arms/METRIC_PROVENANCE.json"
            )
            rows.append(main_row)

    rows.append(
        _row(
            run_id="countdown-v2-postfix-strong-gate-s1",
            suite="countdown",
            experiment="countdown_v2_gate_followup",
            arm="corrected_decay_full_strength_gate",
            seed=1,
            protocol="60 steps; corrected-decay code; single-seed follow-up",
            evidence_path="paper/figures/data/b_strong_gate_1seed.json",
            evidence_locator="complete endpoint summary",
            raw_path=None,
            raw_status="external-not-vendored",
        )
    )
    rows[-1][
        "metric_provenance_path"
    ] = "curriculum_maxrl/countdown_reviewer_arms/METRIC_PROVENANCE.json"

    reviewer_dir = ROOT / "curriculum_maxrl" / "countdown_reviewer_arms"
    verdicts = _json(reviewer_dir / "reviewer_arms_verdicts.json")
    observed_reviewer: dict[str, dict[int, tuple[float, float]]] = {
        "A": {},
        "B": {},
    }
    for path in sorted(reviewer_dir.glob("arm[AB]*_s*.json")):
        match = re.fullmatch(r"arm([AB])_(.+)_s(\d+)\.json", path.name)
        if not match:
            raise ValueError(f"Unexpected reviewer-arm artifact name: {path.name}")
        arm_letter, arm_token, seed_token = match.groups()
        seed = int(seed_token)
        payload = _json(path)
        final_steps = [entry["step"] for entry in payload.get("val", [])]
        if not final_steps or final_steps[-1] != 60:
            raise ValueError(f"{_repo_path(path)} is not a complete step-60 artifact")
        endpoint = payload["val"][-1]
        observed_reviewer[arm_letter][seed] = (
            endpoint["t1_mean16"],
            endpoint["t1_pass16"],
        )
        verdict_key = f"P_R{1 if arm_letter == 'A' else 2}"
        rows.append(
            _row(
                run_id=f"countdown-review-{arm_letter.lower()}-{_slug(arm_token)}-s{seed}",
                suite="countdown",
                experiment="countdown_reviewer_controls",
                arm=(
                    "designed_gate_corrected_decay"
                    if arm_letter == "A"
                    else "higher_dose_live_group_replay_ppo2"
                ),
                seed=seed,
                protocol="preregistered 60-step run; step-60 evaluation; 16 samples/prompt",
                evidence_path=_repo_path(path),
                evidence_locator="complete per-seed artifact",
                raw_path=_repo_path(path),
                raw_status="vendored",
                n_eval_rows=len(payload["val"]),
                verdict=verdicts[verdict_key]["verdict"],
            )
        )
        rows[-1][
            "metric_provenance_path"
        ] = "curriculum_maxrl/countdown_reviewer_arms/METRIC_PROVENANCE.json"

    for arm_letter, verdict_key in (("A", "P_R1"), ("B", "P_R2")):
        per_seed = observed_reviewer[arm_letter]
        if sorted(per_seed) != [1, 2, 3]:
            raise ValueError(
                f"Reviewer arm {arm_letter} has seeds {sorted(per_seed)}, expected 1--3"
            )
        observed_mean = [per_seed[seed][0] for seed in (1, 2, 3)]
        observed_pass = [per_seed[seed][1] for seed in (1, 2, 3)]
        if observed_mean != verdicts[verdict_key]["t1_mean16"]:
            raise ValueError(
                f"{verdict_key} mean endpoints disagree with raw artifacts"
            )
        if observed_pass != verdicts[verdict_key]["t1_pass16"]:
            raise ValueError(
                f"{verdict_key} pass endpoints disagree with raw artifacts"
            )
    return rows


def _gsm8k_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    figure_data_path = ROOT / "paper" / "figures" / "data" / "fig3_gsm8k_data.json"
    figure_data = _json(figure_data_path)

    for seed, key in ((1, "series"), (2, "series_seed2")):
        for arm in sorted(figure_data[key]):
            raw_path: str | None = None
            raw_status = "external-not-vendored"
            evidence_path = _repo_path(figure_data_path)
            evidence_locator = f"{key}.{arm}"
            if seed == 2 and arm == "grpo":
                raw = (
                    ROOT
                    / "curriculum_maxrl"
                    / "gsm8k_artifacts"
                    / "grpo_uniform_seed2.json"
                )
                raw_path = _repo_path(raw)
                raw_status = "vendored"
                evidence_path = raw_path
                evidence_locator = "complete seed-2 uniform-GRPO artifact"
            rows.append(
                _row(
                    run_id=f"gsm8k-original-{_slug(arm)}-s{seed}",
                    suite="gsm8k",
                    experiment="gsm8k_original_factorial",
                    arm=arm,
                    seed=seed,
                    protocol="50 steps; val mean@4 at steps 0, 25, and 50",
                    evidence_path=evidence_path,
                    evidence_locator=evidence_locator,
                    raw_path=raw_path,
                    raw_status=raw_status,
                )
            )

    # The final g3p cell is locally evidenced by the completion audit, but its
    # raw log and stale e_llm1b verdict artifact were never vendored.  We keep
    # the exact registered verdict while leaving unknown seed metadata null.
    g3p_guide_path = ROOT / "FINAL_ICLR_REVIEW_AND_COMPLETION_GUIDE_2026-08-07.md"
    g3p_guide = g3p_guide_path.read_text(encoding="utf-8")
    for required_value in ("0.413", "0.601480", "0.10547", "0.19834"):
        if required_value not in g3p_guide:
            raise ValueError(
                f"{_repo_path(g3p_guide_path)} no longer supports g3p value "
                f"{required_value}"
            )
    g3p = _row(
        run_id="gsm8k-steering-controlled-g3p",
        suite="gsm8k",
        experiment="gsm8k_steering_controlled",
        arm="g3p_power4_teacher_grpo",
        seed=None,
        protocol="pre-registered P-S1; 50 steps; treatment-delivery gate",
        evidence_path=_repo_path(g3p_guide_path),
        evidence_locator="section 5, Latest GSM8K Result",
        raw_path=None,
        raw_status="external-completed-not-vendored",
        verdict="inconclusive-by-design: treatment-delivery mean gate failed",
    )
    g3p["treatment_delivery"] = {
        "minimum_dead_sampled": 0.413,
        "minimum_gate": "<0.50",
        "run_mean_dead_sampled": 0.60148,
        "run_mean_gate": "<0.60",
        "minimum_gate_passed": True,
        "run_mean_gate_passed": False,
    }
    g3p["final_metrics"] = {
        "mean_at_4": 0.10547,
        "pass_at_4": 0.19834,
    }
    rows.append(g3p)
    return rows


def _acrobot_v3_rows() -> list[dict[str, Any]]:
    """Expand the historical 2-arm x 20-seed Acrobot V3 artifact into rows."""
    rows: list[dict[str, Any]] = []
    path = (
        ROOT
        / "frontier_rl"
        / "examples"
        / ("acrobot_neural_v3_shared_confirmatory.json")
    )
    artifact = _json(path)
    protocol = artifact.get("protocol", {})
    expected_seeds = list(range(12_000, 12_020))
    expected_cases = (
        ("uniform_shared_h64", "uniform"),
        ("teacher_shared_h64", "u16_coefficient_mass"),
    )
    if artifact.get("artifact_state") != "complete" or artifact.get("run_failures"):
        raise ValueError(f"{_repo_path(path)} is not a complete artifact")
    if protocol.get("paired_seeds") != expected_seeds:
        raise ValueError(f"{_repo_path(path)} has unexpected paired seeds")
    if protocol.get("n_rollouts") != 16 or protocol.get("budget") != {
        "transition_budget": 2_000_000,
        "optimizer_update_budget": None,
        "transition_safety_cap": None,
    }:
        raise ValueError(f"{_repo_path(path)} has unexpected budget or group size")
    if tuple(artifact.get("cases", {})) != tuple(name for name, _ in expected_cases):
        raise ValueError(f"{_repo_path(path)} has unexpected cases or ordering")

    for case_name, arm in expected_cases:
        case = artifact["cases"][case_name]
        config = case.get("config", {})
        expected_sampling = "uniform" if arm == "uniform" else "teacher"
        if (
            config.get("sampling") != expected_sampling
            or config.get("architecture") != "shared"
            or config.get("hidden_size") != 64
            or config.get("learning_rate") != 3e-4
            or config.get("hindsight_scale") != 0.0
        ):
            raise ValueError(f"{case_name} configuration changed")
        runs = case.get("runs", [])
        if [run.get("seed") for run in runs] != expected_seeds:
            raise ValueError(f"{case_name} seed order changed")
        for index, run in enumerate(runs):
            if not all(
                run.get(key) is True
                for key in (
                    "numeric_valid",
                    "accounting_valid",
                    "verifier_relabel_checks_valid",
                    "evaluation_cadence_invariant",
                )
            ):
                raise ValueError(f"{case_name} seed {run.get('seed')} is invalid")
            if (
                run.get("relabeled_groups") != 0
                or len(run.get("group_diagnostics", [])) != run.get("sampled_groups")
                or len(run.get("x_transitions", [])) < 2
                or run["x_transitions"][-1] != run.get("transitions")
            ):
                raise ValueError(
                    f"{case_name} seed {run.get('seed')} raw accounting changed"
                )
            row = _row(
                run_id=f"acrobot-v3-{_slug(arm)}-s{run['seed']}",
                suite="acrobot",
                experiment="acrobot_v3_shared_curriculum_historical",
                arm=arm,
                seed=int(run["seed"]),
                protocol=(
                    "Acrobot-v1; 8 nested thresholds; practical MaxRL N=16; "
                    "shared H64; no hindsight; 2M nominal actual transitions; "
                    "later audit found neighboring cross-domain RNG-root reuse"
                ),
                evidence_path=_repo_path(path),
                evidence_locator=f"cases.{case_name}.runs[{index}]",
                raw_path=_repo_path(path),
                raw_status="vendored-aggregate-run-record",
                n_eval_rows=len(run["x_transitions"]),
                verdict="historical/descriptive; clean paired inference withdrawn",
            )
            row["raw_locator"] = (
                f"cases.{case_name}.runs[{index}].group_diagnostics and curves"
            )
            row["derived_analysis_path"] = (
                "frontier_rl/examples/acrobot_v3_mechanism_audit.json"
            )
            row["inference_status"] = (
                "cross-seed RNG-domain overlap; retain run record and descriptive "
                "effect only"
            )
            rows.append(row)
    return rows


def _acrobot_tournament_v2_rows() -> list[dict[str, Any]]:
    """Expand only the source-locked V2 development and confirmation runs."""
    rows: list[dict[str, Any]] = []
    lock_path = ROOT / ACROBOT_TOURNAMENT_LOCK
    development_path = ROOT / ACROBOT_TOURNAMENT_DEVELOPMENT
    gate_path = ROOT / ACROBOT_TOURNAMENT_GATE
    confirmatory_path = ROOT / ACROBOT_TOURNAMENT_CONFIRMATORY
    analysis_path = ROOT / ACROBOT_TOURNAMENT_ANALYSIS

    lock = _json(lock_path)
    development = _json(development_path)
    gate = _json(gate_path)
    confirmatory = _json(confirmatory_path)
    analysis = _json(analysis_path)
    lock_sha256 = _sha256(lock_path)
    development_sha256 = _sha256(development_path)
    gate_sha256 = _sha256(gate_path)
    confirmatory_sha256 = _sha256(confirmatory_path)

    condition_names = [case_name for case_name, _, _ in ACROBOT_TOURNAMENT_ARMS]
    development_seeds = list(range(20_100, 20_103))
    confirmatory_seeds = list(range(20_000, 20_020))
    schedule = lock.get("schedule", {})
    expected_schedule = {
        "condition_names": condition_names,
        "confirmatory_seeds": confirmatory_seeds,
        "development_seeds": development_seeds,
        "quick_seeds": [20_200],
        "transition_budget": 2_000_000,
        "eval_interval_transitions": 100_000,
        "eval_n_shared_trajectories": 32,
        "development_transition_budget": 200_000,
        "development_eval_interval_transitions": 50_000,
        "development_eval_n_shared_trajectories": 16,
        "n_rollouts": 16,
    }
    if (
        lock.get("schema") != "curriculum-maxrl/acrobot-curriculum-tournament-lock/v2"
        or lock.get("status") != "sealed_before_v2_development_or_confirmation"
    ):
        raise ValueError(f"{ACROBOT_TOURNAMENT_LOCK} is not the sealed V2 lock")
    for key, expected in expected_schedule.items():
        if schedule.get(key) != expected:
            raise ValueError(f"{ACROBOT_TOURNAMENT_LOCK} has unexpected schedule.{key}")

    expected_lock_binding = {
        "source_lock_relative_path": ACROBOT_TOURNAMENT_LOCK,
        "source_lock_sha256": lock_sha256,
        "source_lock_enforced": True,
    }
    for artifact_path, artifact in (
        (development_path, development),
        (confirmatory_path, confirmatory),
    ):
        provenance = artifact.get("provenance", {})
        if any(
            provenance.get(key) != value for key, value in expected_lock_binding.items()
        ):
            raise ValueError(
                f"{_repo_path(artifact_path)} has an invalid source-lock binding"
            )
        if provenance.get("source_sha256") != lock.get("source_sha256"):
            raise ValueError(
                f"{_repo_path(artifact_path)} source hashes differ from lock"
            )

    required_gate_names = lock.get("development_gate", {}).get(
        "required_checks_in_order"
    )
    if (
        gate.get("schema")
        != "curriculum-maxrl/acrobot-curriculum-tournament-development-gates/v2"
        or gate.get("mode") != "development"
        or gate.get("all_gates_passed") is not True
        or list(gate.get("gates", {})) != required_gate_names
        or not all(value is True for value in gate.get("gates", {}).values())
        or gate.get("gate_policy") != lock.get("development_gate", {}).get("policy")
        or gate.get("raw_artifact_relative_path") != ACROBOT_TOURNAMENT_DEVELOPMENT
        or gate.get("raw_artifact_sha256") != development_sha256
        or gate.get("source_lock_relative_path") != ACROBOT_TOURNAMENT_LOCK
        or gate.get("source_lock_sha256") != lock_sha256
        or gate.get("source_lock_verification", {}).get("passed") is not True
        or gate.get("source_lock_verification", {}).get("source_lock_sha256")
        != lock_sha256
    ):
        raise ValueError(f"{ACROBOT_TOURNAMENT_GATE} has an invalid V2 binding")

    expected_gate_binding = {
        "relative_path": ACROBOT_TOURNAMENT_GATE,
        "sha256": gate_sha256,
        "raw_artifact_relative_path": ACROBOT_TOURNAMENT_DEVELOPMENT,
        "raw_artifact_sha256": development_sha256,
        "all_gates_passed": True,
    }
    if (
        confirmatory.get("protocol", {}).get("development_gate")
        != expected_gate_binding
    ):
        raise ValueError(
            f"{ACROBOT_TOURNAMENT_CONFIRMATORY} is not bound to the passing gate"
        )

    expected_analysis_gate_binding = {
        "passed": True,
        "development_gate_relative_path": ACROBOT_TOURNAMENT_GATE,
        "development_gate_sha256": gate_sha256,
        "development_raw_relative_path": ACROBOT_TOURNAMENT_DEVELOPMENT,
        "development_raw_sha256": development_sha256,
        "gates_recomputed_from_raw": True,
    }
    if (
        analysis.get("schema")
        != "curriculum-maxrl/acrobot-curriculum-tournament-analysis/v2"
        or analysis.get("mode") != "confirmatory"
        or analysis.get("all_checks_passed") is not True
        or analysis.get("raw_artifact_relative_path") != ACROBOT_TOURNAMENT_CONFIRMATORY
        or analysis.get("raw_artifact_sha256") != confirmatory_sha256
        or analysis.get("source_lock_relative_path") != ACROBOT_TOURNAMENT_LOCK
        or analysis.get("source_lock_sha256") != lock_sha256
        or analysis.get("source_lock", {}).get("passed") is not True
        or analysis.get("source_lock", {}).get("source_lock_sha256") != lock_sha256
        or analysis.get("development_gate_verification")
        != expected_analysis_gate_binding
    ):
        raise ValueError(f"{ACROBOT_TOURNAMENT_ANALYSIS} has an invalid V2 binding")

    modes = (
        (
            "development",
            "development_only",
            development_path,
            development,
            development_seeds,
            "development_engine_master_seeds",
            200_000,
            50_000,
            16,
        ),
        (
            "confirmatory",
            "confirmatory",
            confirmatory_path,
            confirmatory,
            confirmatory_seeds,
            "confirmatory_engine_master_seeds",
            2_000_000,
            100_000,
            32,
        ),
    )
    for (
        mode,
        expected_status,
        artifact_path,
        artifact,
        expected_seeds,
        engine_seed_schedule_key,
        transition_budget,
        eval_interval,
        eval_trajectories,
    ) in modes:
        protocol = artifact.get("protocol", {})
        if (
            artifact.get("artifact_state") != "complete"
            or artifact.get("run_failures") != []
        ):
            raise ValueError(
                f"{_repo_path(artifact_path)} is not complete and failure-free"
            )
        if (
            artifact.get("schema")
            != "curriculum-maxrl/acrobot-curriculum-tournament-raw/v2"
            or protocol.get("study") != "acrobot_curriculum_tournament"
            or protocol.get("mode") != mode
            or protocol.get("status") != expected_status
            or protocol.get("condition_names") != condition_names
            or protocol.get("paired_seeds") != expected_seeds
            or list(artifact.get("cases", {})) != condition_names
            or protocol.get("n_rollouts") != 16
            or protocol.get("transition_budget") != transition_budget
            or protocol.get("complete_final_group") is not True
            or protocol.get("eval_interval_transitions") != eval_interval
            or protocol.get("eval_n_shared_trajectories") != eval_trajectories
            or protocol.get("raw_only") is not True
            or protocol.get("logical_to_engine_master_seed")
            != schedule.get(engine_seed_schedule_key)
            or protocol.get("rng_domain_contract", {}).get("rng_domain_offsets")
            != schedule.get("rng_domain_offsets")
            or protocol.get("rng_domain_contract", {}).get(
                "environment_adapter_seed_offset"
            )
            != schedule.get("environment_adapter_seed_offset")
        ):
            raise ValueError(
                f"{_repo_path(artifact_path)} has unexpected mode, arms, seeds, or budgets"
            )

        engine_seed_map = protocol["logical_to_engine_master_seed"]
        rng_offsets = protocol["rng_domain_contract"]["rng_domain_offsets"]
        for case_name, arm, expected_sampling in ACROBOT_TOURNAMENT_ARMS:
            case = artifact["cases"][case_name]
            config = case.get("config", {})
            summary = case.get("summary", {})
            runs = case.get("runs", [])
            if (
                config.get("name") != case_name
                or config.get("stage") != "tournament"
                or config.get("sampling") != expected_sampling
                or config.get("architecture") != "shared"
                or config.get("hidden_size") != 64
                or config.get("learning_rate") != 3e-4
                or config.get("hindsight_scale") != 0.0
                or case.get("sampler")
                != {
                    "uniform": "constant target-uniform 1/8",
                    "p1mp": "p(1-p)",
                    "u16": "1-(1-p)^16-p",
                }[arm]
                or summary.get("n_attempted") != len(expected_seeds)
                or summary.get("n_failed") != 0
                or summary.get("n_valid") != len(expected_seeds)
            ):
                raise ValueError(
                    f"{_repo_path(artifact_path)} case {case_name} changed"
                )
            if [run.get("logical_seed") for run in runs] != expected_seeds:
                raise ValueError(f"{case_name} logical-seed order changed in {mode}")
            if [run.get("seed") for run in runs] != expected_seeds:
                raise ValueError(
                    f"{case_name} seed field is not the logical seed in {mode}"
                )

            for index, (logical_seed, run) in enumerate(zip(expected_seeds, runs)):
                groups = run.get("group_diagnostics", [])
                checkpoints = run.get("checkpoint_records", [])
                expected_master_seed = engine_seed_map.get(str(logical_seed))
                if expected_master_seed is None:
                    raise ValueError(
                        f"{mode} logical seed {logical_seed} lacks a master seed"
                    )
                expected_roots = {
                    domain: expected_master_seed + offset
                    for domain, offset in rng_offsets.items()
                }
                if (
                    run.get("engine_master_seed") != expected_master_seed
                    or run.get("environment_adapter_seed_argument")
                    != expected_master_seed + 1_000
                    or run.get("rng_roots") != expected_roots
                ):
                    raise ValueError(
                        f"{case_name} logical seed {logical_seed} RNG binding changed"
                    )
                if not all(
                    run.get(key) is True
                    for key in (
                        "numeric_valid",
                        "accounting_valid",
                        "verifier_relabel_checks_valid",
                        "evaluation_cadence_invariant",
                    )
                ) or not all(run.get("evaluation_rng_preserved", [])):
                    raise ValueError(
                        f"{case_name} logical seed {logical_seed} is invalid"
                    )

                task_groups = run.get("task_groups", [])
                task_rollouts = run.get("task_rollouts", [])
                task_transitions = run.get("task_transitions", [])
                observed_task_groups = Counter(group.get("task_id") for group in groups)
                if (
                    run.get("relabeled_groups") != 0
                    or run.get("sampled_groups") != len(groups)
                    or run.get("rollout_attempts") != 16 * len(groups)
                    or len(task_groups) != 8
                    or len(task_rollouts) != 8
                    or len(task_transitions) != 8
                    or task_groups
                    != [observed_task_groups.get(task_id, 0) for task_id in range(8)]
                    or any(
                        rollouts != 16 * group_count
                        for rollouts, group_count in zip(task_rollouts, task_groups)
                    )
                    or sum(task_rollouts) != run.get("rollout_attempts")
                    or sum(task_transitions) != run.get("transitions")
                    or [group.get("group") for group in groups]
                    != list(range(1, len(groups) + 1))
                    or run.get("total_parameters") != 640
                    or run.get("active_parameters_per_task") != 640
                ):
                    raise ValueError(
                        f"{case_name} logical seed {logical_seed} accounting changed"
                    )
                previous_end = 0
                for group in groups:
                    if (
                        group.get("transition_start") != previous_end
                        or group.get("transition_end")
                        != previous_end + group.get("n_transitions", -1)
                        or not 0 <= group.get("success_count", -1) <= 16
                        or not 0 < group.get("n_transitions", 0) <= 8_000
                    ):
                        raise ValueError(
                            f"{case_name} logical seed {logical_seed} group ledger changed"
                        )
                    previous_end = group["transition_end"]
                if not groups:
                    raise ValueError(
                        f"{case_name} logical seed {logical_seed} has no groups"
                    )

                final_group = groups[-1]
                transitions = run.get("transitions")
                x_transitions = run.get("x_transitions", [])
                expected_eval_rows = transition_budget // eval_interval + 1
                expected_eval_targets = [
                    checkpoint * eval_interval
                    for checkpoint in range(expected_eval_rows)
                ]
                if (
                    final_group.get("transition_start") >= transition_budget
                    or final_group.get("transition_end") != transitions
                    or not transition_budget < transitions <= transition_budget + 8_000
                    or run.get("transition_cap_censored") is not False
                    or len(x_transitions) != expected_eval_rows
                    or any(
                        not target <= observed <= target + 8_000
                        for target, observed in zip(
                            expected_eval_targets, x_transitions
                        )
                    )
                    or x_transitions[-1] != transitions
                    or [checkpoint.get("transitions") for checkpoint in checkpoints]
                    != x_transitions
                    or any(
                        checkpoint.get("evaluation_shared_trajectories")
                        != eval_trajectories
                        or len(checkpoint.get("pass_rates", [])) != 8
                        for checkpoint in checkpoints
                    )
                    or len(run.get("evaluation_rng_preserved", []))
                    != len(x_transitions)
                    or len(run.get("pass_rate_curve", [])) != len(x_transitions)
                ):
                    raise ValueError(
                        f"{case_name} logical seed {logical_seed} final-group "
                        "overshoot or evaluation accounting changed"
                    )

                row = _row(
                    run_id=f"acrobot-tournament-v2-{mode}-{arm}-s{logical_seed}",
                    suite="acrobot",
                    experiment=f"acrobot_curriculum_tournament_v2_{mode}",
                    arm=arm,
                    seed=int(run["logical_seed"]),
                    protocol=(
                        f"source-locked V2 {mode}; Acrobot-v1; practical MaxRL "
                        f"N=16; {transition_budget:,}-transition crossing target; "
                        "complete final group"
                    ),
                    evidence_path=_repo_path(artifact_path),
                    evidence_locator=f"cases.{case_name}.runs[{index}]",
                    raw_path=_repo_path(artifact_path),
                    raw_status="vendored-aggregate-run-record",
                    n_eval_rows=len(x_transitions),
                )
                row["mode"] = mode
                row["raw_locator"] = f"cases.{case_name}.runs[{index}]"
                row["source_lock_path"] = ACROBOT_TOURNAMENT_LOCK
                if mode == "confirmatory":
                    row["development_gate_path"] = ACROBOT_TOURNAMENT_GATE
                    row["derived_analysis_path"] = ACROBOT_TOURNAMENT_ANALYSIS
                rows.append(row)

    expected_row_order = [
        (mode, arm, logical_seed)
        for mode, seeds in (
            ("development", development_seeds),
            ("confirmatory", confirmatory_seeds),
        )
        for _, arm, _ in ACROBOT_TOURNAMENT_ARMS
        for logical_seed in seeds
    ]
    actual_row_order = [(row["mode"], row["arm"], row["seed"]) for row in rows]
    if actual_row_order != expected_row_order or len(rows) != 69:
        raise ValueError("V2 Acrobot registry row mode/arm/logical-seed order changed")
    return rows


def _acrobot_procurl_selection_rows() -> list[dict[str, Any]]:
    """Expand the audited development and confirmation ProCuRL-selection runs.

    The development aggregate is vendored.  The 1.37 GB confirmatory aggregate
    is deliberately external: each registry row instead binds its vendored
    descriptive record to the content-addressed per-run entry in the external
    raw manifest.  No machine-local path or unadvertised download location is
    serialized.
    """

    paths = {
        relative: ROOT / relative
        for relative in ACROBOT_PROCURL_EXPECTED_SHA256
    }
    missing = [relative for relative, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"missing Acrobot ProCuRL artifacts: {missing}")
    for relative, expected_sha256 in ACROBOT_PROCURL_EXPECTED_SHA256.items():
        observed_sha256 = _sha256(paths[relative])
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"{relative} changed: {observed_sha256} != {expected_sha256}"
            )
    lock = _json(paths[ACROBOT_PROCURL_LOCK])
    development = _json(paths[ACROBOT_PROCURL_DEVELOPMENT])
    gate = _json(paths[ACROBOT_PROCURL_GATE])
    analysis = _json(paths[ACROBOT_PROCURL_ANALYSIS])
    portable = _json(paths[ACROBOT_PROCURL_PORTABLE])
    diagnostics = _json(paths[ACROBOT_PROCURL_DIAGNOSTICS])
    external_manifest = _json(paths[ACROBOT_PROCURL_EXTERNAL_MANIFEST])

    arm_names = [name for name, _, _ in ACROBOT_PROCURL_ARMS]
    development_seeds = [21_300, 21_301, 21_302]
    confirmatory_seeds = list(range(21_000, 21_080))
    expected_schedule = {
        "arm_names": arm_names,
        "confirmatory_seeds": confirmatory_seeds,
        "development_seeds": development_seeds,
        "quick_seeds": [21_400],
        "confirmatory_paid_budget": 2_000_000,
        "development_paid_budget": 400_000,
        "quick_paid_budget": 100_000,
        "regular_eval_interval_paid": 100_000,
        "confirmatory_eval_n": 32,
        "development_eval_n": 32,
        "quick_eval_n": 2,
        "n_rollouts": 16,
        "learning_rate": 3e-4,
        "probes_per_task": 20,
        "refresh_student_transitions": 5_120,
        "procurl_beta": 20.0,
        "u16_beta_continuous_range_matched": 6.416133525771289,
        "u16_lattice_max_logit": 4.97730861318145,
        "engine_master_base": 50_000_000_000,
        "engine_master_stride": 10_000_000,
        "rng_domain_offsets": {
            "actor_parameter": 0,
            "actor_action": 1,
            "selection": 10_000,
            "environment_reset_rng": 11_003,
            "evaluation_episode": 1_000_000,
            "evaluation_action": 1_000_001,
            "probe_episode_reset": 2_000_000,
            "probe_episode_action": 3_000_000,
        },
        "environment_adapter_seed_offset": 1_000,
        "upstream_procurl_commit": (
            "17904f1d7b9b29e089d4f70ae7aadf1da50ba6b2"
        ),
    }
    if (
        lock.get("schema")
        != "curriculum-maxrl/acrobot-procurl-selection-lock/v1"
        or lock.get("status")
        != "sealed_before_any_quick_development_or_confirmation"
        or lock.get("schedule") != expected_schedule
    ):
        raise ValueError(f"{ACROBOT_PROCURL_LOCK} is not the audited sealed lock")

    lock_sha256 = ACROBOT_PROCURL_EXPECTED_SHA256[ACROBOT_PROCURL_LOCK]
    development_sha256 = ACROBOT_PROCURL_EXPECTED_SHA256[
        ACROBOT_PROCURL_DEVELOPMENT
    ]
    gate_sha256 = ACROBOT_PROCURL_EXPECTED_SHA256[ACROBOT_PROCURL_GATE]
    analysis_sha256 = ACROBOT_PROCURL_EXPECTED_SHA256[ACROBOT_PROCURL_ANALYSIS]
    portable_sha256 = ACROBOT_PROCURL_EXPECTED_SHA256[ACROBOT_PROCURL_PORTABLE]

    development_protocol = development.get("protocol", {})
    development_provenance = development.get("provenance", {})
    if (
        development.get("schema")
        != "curriculum-maxrl/acrobot-procurl-selection-raw/v1"
        or development.get("artifact_state") != "complete"
        or development.get("run_failures") != []
        or development_protocol.get("study")
        != "acrobot_procurl_selection_semantics"
        or development_protocol.get("mode") != "development"
        or development_protocol.get("arm_names") != arm_names
        or development_protocol.get("paired_logical_seeds") != development_seeds
        or development_protocol.get("paid_budget_nominal") != 400_000
        or development_protocol.get("student_group_size") != 16
        or development_provenance.get("source_lock_relative_path")
        != ACROBOT_PROCURL_LOCK
        or development_provenance.get("source_lock_sha256") != lock_sha256
        or development_provenance.get("source_lock_enforced") is not True
        or development_provenance.get("source_sha256") != lock.get("source_sha256")
        or list(development.get("cases", {})) != arm_names
    ):
        raise ValueError(f"{ACROBOT_PROCURL_DEVELOPMENT} binding or schedule changed")

    required_gate_names = (
        "all_runs_source_numeric_parameter_rng_ledger_valid",
        "all_sweeps_exact_probe_count_and_bounded_transitions",
        "all_p_hat_values_are_multiples_of_0p05",
        "initial_and_crossed_boundary_sweep_schedule_exact",
        "probes_preserve_actor_optimizer_and_training_rng",
        "paid_equals_student_plus_probe",
        "uniform_arms_exact_and_ordinary_has_no_probes",
        "adaptive_probabilities_recompute_and_nonuniform_once",
        "each_probed_run_has_20k_student_transitions_and_update",
        "pooled_dead_mixed_all_pass_regimes_observed",
        "pooled_native_evaluation_values_vary",
    )
    if (
        gate.get("schema")
        != "curriculum-maxrl/acrobot-procurl-selection-development-gates/v1"
        or gate.get("mode") != "development"
        or gate.get("all_gates_passed") is not True
        or tuple(gate.get("gates", {})) != required_gate_names
        or not all(value is True for value in gate.get("gates", {}).values())
        or gate.get("source_lock_sha256") != lock_sha256
        or gate.get("raw_artifact_relative_path") != ACROBOT_PROCURL_DEVELOPMENT
        or gate.get("raw_artifact_sha256") != development_sha256
        or gate.get("source_lock_verification", {}).get("passed") is not True
        or gate.get("diagnostics", {}).get("n_runs") != 12
        or gate.get("diagnostics", {}).get("arm_contrasts_computed") is not False
        or gate.get("gate_policy", {}).get("outcome_blind") is not True
        or gate.get("gate_policy", {}).get("uses_arm_contrasts") is not False
    ):
        raise ValueError(f"{ACROBOT_PROCURL_GATE} is not the passing outcome-blind gate")

    primary = analysis.get("primary", {})
    expected_secondary_decisions = {
        "procurl_minus_sham": False,
        "u16_minus_sham": False,
        "procurl_minus_ordinary": True,
        "u16_minus_ordinary": True,
        "sham_minus_ordinary": True,
    }
    secondary = analysis.get("secondary_holm_family", {})
    if (
        analysis.get("schema")
        != "curriculum-maxrl/acrobot-procurl-selection-analysis/v1"
        or analysis.get("mode") != "confirmatory"
        or analysis.get("strict_validation_passed") is not True
        or analysis.get("raw_artifact_relative_path") != ACROBOT_PROCURL_EXTERNAL_RAW
        or analysis.get("raw_artifact_sha256")
        != ACROBOT_PROCURL_EXTERNAL_RAW_SHA256
        or analysis.get("source_lock_verification", {}).get("passed") is not True
        or analysis.get("source_lock_verification", {}).get("source_lock_sha256")
        != lock_sha256
        or analysis.get("development_gate_binding_verification", {}).get("passed")
        is not True
        or analysis.get("development_gate_binding_verification", {}).get(
            "development_gate_sha256"
        )
        != gate_sha256
        or primary.get("left") != "u16_probe_range_matched_f5120"
        or primary.get("right") != "procurl_env_b20_f5120"
        or primary.get("n_pairs") != 80
        or primary.get("mean_contrast") != 0.004894235861048817
        or primary.get("paired_t_p_two_sided") != 0.05149237843697304
        or primary.get("sesoi") != 0.02
        or primary.get("supported") is not False
        or list(secondary) != list(expected_secondary_decisions)
        or any(
            secondary[name].get("reject_familywise_0.05") is not expected
            for name, expected in expected_secondary_decisions.items()
        )
    ):
        raise ValueError(f"{ACROBOT_PROCURL_ANALYSIS} audited result changed")

    if (
        portable.get("schema")
        != "curriculum-maxrl/acrobot-procurl-selection-portable-verification/v1"
        or portable.get("all_checks_passed") is not True
        or portable.get("source_lock_sha256") != lock_sha256
        or portable.get("source_manifest_verification", {}).get("passed") is not True
        or portable.get("live_reanalysis_runtime_verification", {}).get("passed")
        is not True
        or portable.get("invalid_pre_gate_archive_verification", {}).get("passed")
        is not True
        or portable.get("raw_ledger_validation", {}).get("passed") is not True
        or portable.get("raw_ledger_validation", {}).get("paired_seed_count") != 80
        or portable.get("raw_ledger_validation", {}).get("arm_count") != 4
        or portable.get("development_gate_binding_verification", {}).get("passed")
        is not True
        or portable.get("stored_analysis_comparison", {}).get("passed") is not True
        or portable.get("stored_analysis_comparison", {}).get(
            "stored_analysis_sha256"
        )
        != analysis_sha256
    ):
        raise ValueError(f"{ACROBOT_PROCURL_PORTABLE} verification receipt changed")

    raw_binding = external_manifest.get("raw_artifact", {})
    manifest_schedule = external_manifest.get("schedule", {})
    expected_manifest_bindings = {
        "source_lock": (
            ACROBOT_PROCURL_LOCK,
            lock_sha256,
            "curriculum-maxrl/acrobot-procurl-selection-lock/v1",
        ),
        "development_gate": (
            ACROBOT_PROCURL_GATE,
            gate_sha256,
            "curriculum-maxrl/acrobot-procurl-selection-development-gates/v1",
        ),
        "confirmatory_analysis": (
            ACROBOT_PROCURL_ANALYSIS,
            analysis_sha256,
            "curriculum-maxrl/acrobot-procurl-selection-analysis/v1",
        ),
        "portable_verification": (
            ACROBOT_PROCURL_PORTABLE,
            portable_sha256,
            "curriculum-maxrl/acrobot-procurl-selection-portable-verification/v1",
        ),
    }
    if (
        external_manifest.get("schema")
        != "curriculum-maxrl/acrobot-procurl-selection-external-raw-manifest/v1"
        or external_manifest.get("study")
        != "acrobot_procurl_selection_semantics"
        or external_manifest.get("mode") != "confirmatory"
        or raw_binding
        != {
            "logical_path": ACROBOT_PROCURL_EXTERNAL_RAW,
            "size_bytes": ACROBOT_PROCURL_EXTERNAL_RAW_SIZE,
            "sha256": ACROBOT_PROCURL_EXTERNAL_RAW_SHA256,
            "schema": "curriculum-maxrl/acrobot-procurl-selection-raw/v1",
        }
        or manifest_schedule.get("arms") != arm_names
        or manifest_schedule.get("seeds") != confirmatory_seeds
        or manifest_schedule.get("run_count") != 320
        or manifest_schedule.get("index_order")
        != "arm-major-then-frozen-seed-order"
        or list(external_manifest.get("bindings", {}))
        != list(expected_manifest_bindings)
    ):
        raise ValueError(f"{ACROBOT_PROCURL_EXTERNAL_MANIFEST} changed")
    for role, (logical_path, sha256, schema) in expected_manifest_bindings.items():
        binding = external_manifest["bindings"][role]
        if binding != {
            "logical_path": logical_path,
            "size_bytes": (ROOT / logical_path).stat().st_size,
            "sha256": sha256,
            "schema": schema,
        }:
            raise ValueError(f"external manifest binding {role} changed")

    if (
        diagnostics.get("schema")
        != "curriculum-maxrl/acrobot-procurl-selection-descriptive-diagnostics/v1"
        or diagnostics.get("mode") != "confirmatory"
        or diagnostics.get("status") != "descriptive_only_no_new_inference"
        or diagnostics.get("raw_artifact") != raw_binding
        or diagnostics.get("source_lock")
        != external_manifest["bindings"]["source_lock"]
        or diagnostics.get("development_gate")
        != external_manifest["bindings"]["development_gate"]
        or diagnostics.get("schedule")
        != {
            "arms": arm_names,
            "seeds": confirmatory_seeds,
            "run_count": 320,
        }
        or diagnostics.get("metric_policy", {}).get("new_inferential_statistics")
        is not False
        or list(diagnostics.get("arms", {})) != arm_names
    ):
        raise ValueError(f"{ACROBOT_PROCURL_DIAGNOSTICS} binding changed")

    rows: list[dict[str, Any]] = []
    for case_name, expected_selection, expected_probes in ACROBOT_PROCURL_ARMS:
        case = development["cases"][case_name]
        config = case.get("config", {})
        runs = case.get("runs", [])
        summary = case.get("summary", {})
        if (
            config
            != {
                "name": case_name,
                "selection": expected_selection,
                "probes": expected_probes,
            }
            or summary.get("n_attempted") != 3
            or summary.get("n_valid") != 3
            or summary.get("n_failed") != 0
            or [run.get("logical_seed") for run in runs] != development_seeds
            or [run.get("seed") for run in runs] != development_seeds
        ):
            raise ValueError(f"development case {case_name} changed")
        for index, run in enumerate(runs):
            if (
                run.get("numeric_valid") is not True
                or run.get("accounting_valid") is not True
                or run.get("probe_training_state_preserved") is not True
                or run.get("evaluation_rng_preserved") is not True
                or run.get("paid_transitions")
                != run.get("student_transitions") + run.get("probe_transitions")
                or run.get("probe_sweeps")
                != len(run.get("probe_sweep_records", []))
                or not isinstance(run.get("evaluation_records"), list)
            ):
                raise ValueError(
                    f"development {case_name} seed {run.get('seed')} is invalid"
                )
            row = _row(
                run_id=(
                    f"acrobot-procurl-selection-development-{_slug(case_name)}-"
                    f"s{run['logical_seed']}"
                ),
                suite="acrobot",
                experiment="acrobot_procurl_selection_semantics_development",
                arm=case_name,
                seed=int(run["logical_seed"]),
                protocol=(
                    "source-locked selection semantics on fixed MaxRL Acrobot learner; "
                    "400k nominal paid transitions; outcome-blind development only"
                ),
                evidence_path=ACROBOT_PROCURL_DEVELOPMENT,
                evidence_locator=f"cases.{case_name}.runs[{index}]",
                raw_path=ACROBOT_PROCURL_DEVELOPMENT,
                raw_status="vendored-aggregate-run-record",
                n_eval_rows=len(run["evaluation_records"]),
                verdict="development-gate-valid; excluded from confirmatory contrasts",
            )
            row["mode"] = "development"
            row["raw_locator"] = f"cases.{case_name}.runs[{index}]"
            row["source_lock_path"] = ACROBOT_PROCURL_LOCK
            row["development_gate_path"] = ACROBOT_PROCURL_GATE
            row["results_path"] = ACROBOT_PROCURL_RESULTS
            rows.append(row)

    run_index = external_manifest.get("run_index", [])
    expected_index = [
        (ordinal, arm, seed)
        for ordinal, (arm, seed) in enumerate(
            (arm, seed) for arm in arm_names for seed in confirmatory_seeds
        )
    ]
    observed_index = [
        (entry.get("ordinal"), entry.get("arm"), entry.get("seed"))
        for entry in run_index
    ]
    if observed_index != expected_index:
        raise ValueError("external raw run index order or membership changed")

    for case_name in arm_names:
        arm_diagnostics = diagnostics["arms"][case_name]
        per_seed = arm_diagnostics.get("per_seed", [])
        if (
            [record.get("seed") for record in per_seed] != confirmatory_seeds
            or len(per_seed) != 80
            or not isinstance(arm_diagnostics.get("descriptive_summary"), dict)
        ):
            raise ValueError(f"confirmatory diagnostics for {case_name} changed")
        for index, record in enumerate(per_seed):
            ordinal = arm_names.index(case_name) * len(confirmatory_seeds) + index
            raw_record = run_index[ordinal]
            raw_run_sha256 = raw_record.get("canonical_json_sha256")
            raw_run_size = raw_record.get("canonical_json_size_bytes")
            if (
                not isinstance(raw_run_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", raw_run_sha256)
                or not isinstance(raw_run_size, int)
                or raw_run_size <= 0
            ):
                raise ValueError(f"external raw run index {ordinal} is invalid")
            row = _row(
                run_id=(
                    f"acrobot-procurl-selection-confirmatory-{_slug(case_name)}-"
                    f"s{record['seed']}"
                ),
                suite="acrobot",
                experiment="acrobot_procurl_selection_semantics_confirmatory",
                arm=case_name,
                seed=int(record["seed"]),
                protocol=(
                    "source-locked selection semantics on fixed MaxRL Acrobot learner; "
                    "2M nominal paid transitions; 20 probes/task every 5120 student "
                    "transitions for probed arms"
                ),
                evidence_path=ACROBOT_PROCURL_DIAGNOSTICS,
                evidence_locator=f"arms.{case_name}.per_seed[{index}]",
                raw_path=None,
                raw_status="external-content-addressed-aggregate-run-record",
                verdict="registered primary not supported",
            )
            row["mode"] = "confirmatory"
            row["source_lock_path"] = ACROBOT_PROCURL_LOCK
            row["development_gate_path"] = ACROBOT_PROCURL_GATE
            row["derived_analysis_path"] = ACROBOT_PROCURL_ANALYSIS
            row["portable_verification_path"] = ACROBOT_PROCURL_PORTABLE
            row["external_raw_manifest_path"] = ACROBOT_PROCURL_EXTERNAL_MANIFEST
            row["results_path"] = ACROBOT_PROCURL_RESULTS
            row["raw_locator"] = f"run_index[{ordinal}]"
            row["raw_artifact_sha256"] = ACROBOT_PROCURL_EXTERNAL_RAW_SHA256
            row["raw_artifact_size_bytes"] = ACROBOT_PROCURL_EXTERNAL_RAW_SIZE
            row["raw_run_sha256"] = raw_run_sha256
            row["raw_run_size_bytes"] = raw_run_size
            row["content_addressed_download_uri"] = None
            row["inference_status"] = (
                "u16-minus-ProCuRL primary unsupported; probe-cost/cadence result "
                "is setting-specific"
            )
            rows.append(row)

    expected_row_order = [
        (mode, arm, seed)
        for mode, seeds in (
            ("development", development_seeds),
            ("confirmatory", confirmatory_seeds),
        )
        for arm in arm_names
        for seed in seeds
    ]
    actual_row_order = [(row["mode"], row["arm"], row["seed"]) for row in rows]
    if actual_row_order != expected_row_order or len(rows) != 332:
        raise ValueError("ProCuRL-selection registry row order or count changed")
    return rows


def _external_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    factorial = [
        r["run_id"] for r in rows if r["experiment"].startswith("balanced_factorial")
    ]
    legacy = [
        r["run_id"]
        for r in rows
        if r["experiment"] == "matched_wall_clock_cohort" and r["raw_path"] is None
    ]
    countdown = [
        r["run_id"] for r in rows if r["suite"] == "countdown" and r["raw_path"] is None
    ]
    gsm8k = [
        r["run_id"] for r in rows if r["suite"] == "gsm8k" and r["raw_path"] is None
    ]
    procurl_confirmatory = [
        r["run_id"]
        for r in rows
        if r["experiment"]
        == "acrobot_procurl_selection_semantics_confirmatory"
    ]
    return [
        {
            "gap_id": "maze-factorial-raw-logs",
            "status": "external-at-execution-fork-9f7dd2e",
            "n_affected_runs": len(factorial),
            "related_run_ids": factorial,
            "evidence_path": "curriculum_maxrl/maze_gpu_factorial/PROVENANCE.md",
            "note": "Per-step fact250 JSONL files and warmstarts are not vendored; per-cell result summaries are vendored.",
        },
        {
            "gap_id": "maze-legacy-cohort-raw-logs",
            "status": "external-not-vendored",
            "n_affected_runs": len(legacy),
            "related_run_ids": legacy,
            "evidence_path": "curriculum_maxrl/un_form_verdicts.json",
            "note": "Eight legacy FrontierMax/UN-form raw logs are absent; per-seed endpoint summaries are vendored.",
        },
        {
            "gap_id": "countdown-main-raw-logs",
            "status": "external-not-vendored",
            "n_affected_runs": len(countdown),
            "related_run_ids": countdown,
            "evidence_path": "COUNTDOWN_ANALYSIS.md",
            "note": "Main-arm and predecessor raw logs are external; aggregates and all six reviewer-control artifacts are vendored.",
        },
        {
            "gap_id": "gsm8k-raw-and-verdict-artifacts",
            "status": "external-not-vendored",
            "n_affected_runs": len(gsm8k),
            "related_run_ids": gsm8k,
            "evidence_path": "FINAL_REVIEW_RESPONSE_AND_GUIDANCE_2026-08-07.md",
            "note": "Most original-cell raw logs, the completed g3p raw log, and the refreshed e_llm1b verdict are absent. g3s pass-2 and its queued controls have no locally evidenced completion and are not counted as completed runs.",
        },
        {
            "gap_id": "acrobot-procurl-selection-confirmatory-raw",
            "status": "external-content-addressed-aggregate-run-record",
            "n_affected_runs": len(procurl_confirmatory),
            "related_run_ids": procurl_confirmatory,
            "evidence_path": ACROBOT_PROCURL_EXTERNAL_MANIFEST,
            "raw_artifact_logical_path": ACROBOT_PROCURL_EXTERNAL_RAW,
            "raw_artifact_size_bytes": ACROBOT_PROCURL_EXTERNAL_RAW_SIZE,
            "raw_artifact_sha256": ACROBOT_PROCURL_EXTERNAL_RAW_SHA256,
            "content_addressed_download_uri": None,
            "note": (
                "The ignored 1.37 GB raw aggregate is retained externally. The "
                "vendored manifest binds all 320 canonical run records and the "
                "compact analysis, portable receipt, lock, and development gate; "
                "no public download URI is currently available."
            ),
        },
    ]


def _not_counted_runs() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": "maze-checkpoint-ck-falp-hsd-s1",
            "status": "legacy-registry-only-no-local-evidence",
            "evidence_path": "curriculum_maxrl/RUN_REGISTRY.md",
            "note": "The registry at commit 0fd5f70 named ck_falp_hsd_s1.jsonl, but no raw artifact, endpoint summary, or current paper use is locally evidenced; it is not counted as complete.",
        },
        {
            "candidate_id": "gsm8k-steering-g3s-pass2",
            "status": "completion-not-locally-evidenced",
            "evidence_path": "FINAL_REVIEW_RESPONSE_AND_GUIDANCE_2026-08-07.md",
            "note": "Last local status records a pass-2 retry in flight and recommends stopping or archiving it after P-S1 was already decided.",
        },
        {
            "candidate_id": "gsm8k-steering-g3u-control",
            "status": "queued-external-status-unknown",
            "evidence_path": "FINAL_REVIEW_RESPONSE_AND_GUIDANCE_2026-08-07.md",
            "note": "No local completion artifact; not interpretable without a treatment-delivered steering cell.",
        },
        {
            "candidate_id": "gsm8k-steering-m3s-control",
            "status": "queued-external-status-unknown",
            "evidence_path": "FINAL_REVIEW_RESPONSE_AND_GUIDANCE_2026-08-07.md",
            "note": "No local completion artifact; not interpretable without a treatment-delivered steering cell.",
        },
    ]


def build_registry() -> dict[str, Any]:
    rows = (
        _maze_cohort_rows()
        + _maze_factorial_rows()
        + _countdown_rows()
        + _gsm8k_rows()
        + _acrobot_v3_rows()
        + _acrobot_tournament_v2_rows()
        + _acrobot_procurl_selection_rows()
    )
    suite_counts = dict(sorted(Counter(row["suite"] for row in rows).items()))
    experiment_counts = dict(sorted(Counter(row["experiment"] for row in rows).items()))
    raw_counts = dict(sorted(Counter(row["raw_status"] for row in rows).items()))
    aggregate_run_records = sum(
        count
        for status, count in raw_counts.items()
        if status.endswith("aggregate-run-record")
    )
    registry = {
        "schema_version": 2,
        "generated": GENERATED_DATE,
        "generator": "curriculum_maxrl/build_run_registry.py",
        "note": (
            "One row per paper-used training run supported by a local artifact "
            "or summary in the maze, Countdown, GSM8K, and Acrobot suites. "
            "Aggregate cells use evidence_locator; external raw artifacts are "
            "explicit, content-addressed when possible, and never represented by "
            "machine-local paths. Protocol labels such as "
            "'preregistered' are transcribed from source evidence records; absent "
            "locking objects mean this registry does not independently establish "
            "their timing."
        ),
        "n_rows": len(rows),
        "counts": {
            "by_suite": suite_counts,
            "by_experiment": experiment_counts,
            "by_raw_status": raw_counts,
            "aggregate_run_records": aggregate_run_records,
        },
        "rows": rows,
        "external_artifact_gaps": _external_gaps(rows),
        "not_counted_runs": _not_counted_runs(),
    }
    validate_registry(registry)
    return registry


def _all_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_strings(child)


def validate_registry(registry: dict[str, Any]) -> None:
    rows = registry["rows"]
    if registry["n_rows"] != len(rows):
        raise ValueError("n_rows does not match rows length")

    ids = [row["run_id"] for row in rows]
    duplicates = sorted(run_id for run_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate run_id values: {duplicates}")

    required = {
        "run_id",
        "suite",
        "experiment",
        "arm",
        "seed",
        "status",
        "protocol",
        "evidence_path",
        "evidence_locator",
        "raw_path",
        "raw_status",
    }
    for row in rows:
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"{row.get('run_id', '<unknown>')} missing {missing}")
        for key, value in row.items():
            if not key.endswith("_path"):
                continue
            if value is None:
                continue
            path = Path(value)
            if path.is_absolute():
                raise ValueError(f"{row['run_id']} contains absolute {key}: {value}")
            if not (ROOT / path).is_file():
                raise ValueError(f"{row['run_id']} references missing {key}: {value}")
        is_vendored = row["raw_status"].startswith("vendored")
        if is_vendored and row["raw_path"] is None:
            raise ValueError(f"{row['run_id']} says vendored but has no raw_path")
        if not is_vendored and row["raw_path"] is not None:
            raise ValueError(f"{row['run_id']} has raw_path but status is not vendored")

    expected_counts = {
        "by_suite": dict(sorted(Counter(row["suite"] for row in rows).items())),
        "by_experiment": dict(
            sorted(Counter(row["experiment"] for row in rows).items())
        ),
        "by_raw_status": dict(
            sorted(Counter(row["raw_status"] for row in rows).items())
        ),
        "aggregate_run_records": sum(
            1
            for row in rows
            if row["raw_status"].endswith("aggregate-run-record")
        ),
    }
    if registry["counts"] != expected_counts:
        raise ValueError("top-level counts are stale")

    expected_suite_counts = {
        "acrobot": 441,
        "countdown": 20,
        "gsm8k": 7,
        "maze": 94,
    }
    if (
        registry["n_rows"] != 562
        or registry["counts"]["by_suite"] != expected_suite_counts
    ):
        raise ValueError("registry total or suite inventory changed")
    raw_counts = registry["counts"]["by_raw_status"]
    if (
        raw_counts.get("vendored-aggregate-run-record") != 121
        or raw_counts.get("external-content-addressed-aggregate-run-record")
        != 320
        or raw_counts.get("vendored") != 33
        or sum(
            count
            for status, count in raw_counts.items()
            if not status.startswith("vendored")
        )
        != 408
        or registry["counts"]["aggregate_run_records"] != 441
    ):
        raise ValueError("registry raw-artifact accounting changed")

    gap_run_ids: list[str] = []
    for gap in registry["external_artifact_gaps"]:
        evidence = Path(gap["evidence_path"])
        if evidence.is_absolute() or not (ROOT / evidence).is_file():
            raise ValueError(f"{gap['gap_id']} has invalid evidence_path")
        if gap["n_affected_runs"] != len(gap["related_run_ids"]):
            raise ValueError(f"{gap['gap_id']} has stale n_affected_runs")
        unknown = sorted(set(gap["related_run_ids"]) - set(ids))
        if unknown:
            raise ValueError(f"{gap['gap_id']} references unknown runs: {unknown}")
        gap_run_ids.extend(gap["related_run_ids"])

    expected_gap_ids = sorted(row["run_id"] for row in rows if row["raw_path"] is None)
    if sorted(gap_run_ids) != expected_gap_ids:
        raise ValueError(
            "external_artifact_gaps must cover each non-vendored run exactly once"
        )

    not_counted_ids: set[str] = set()
    for candidate in registry["not_counted_runs"]:
        candidate_id = candidate["candidate_id"]
        if candidate_id in not_counted_ids or candidate_id in ids:
            raise ValueError(f"Duplicate counted/not-counted ID: {candidate_id}")
        not_counted_ids.add(candidate_id)
        evidence = Path(candidate["evidence_path"])
        if evidence.is_absolute() or not (ROOT / evidence).is_file():
            raise ValueError(f"{candidate_id} has invalid evidence_path")

    # Reject copied EC2/Mac/temp paths anywhere in the generated registry,
    # including prose fields.  Repository-relative paths never start here.
    machine_prefixes = ("/home/", "/Users/", "/tmp/", "/var/folders/")
    leaked = sorted(
        {text for text in _all_strings(registry) if text.startswith(machine_prefixes)}
    )
    if leaked:
        raise ValueError(f"Machine-local absolute paths leaked into registry: {leaked}")


def _serialized(registry: dict[str, Any]) -> str:
    return json.dumps(registry, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate sources and fail if run_registry.json is not current",
    )
    args = parser.parse_args()

    registry = build_registry()
    rendered = _serialized(registry)
    if args.check:
        if not REGISTRY_PATH.exists():
            raise SystemExit(f"missing {_repo_path(REGISTRY_PATH)}")
        if REGISTRY_PATH.read_text(encoding="utf-8") != rendered:
            raise SystemExit(
                "run_registry.json is stale; run "
                "python3 curriculum_maxrl/build_run_registry.py"
            )
    else:
        REGISTRY_PATH.write_text(rendered, encoding="utf-8")

    print(
        f"run registry OK: {registry['n_rows']} runs; "
        f"suites={registry['counts']['by_suite']}; "
        f"raw={registry['counts']['by_raw_status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
