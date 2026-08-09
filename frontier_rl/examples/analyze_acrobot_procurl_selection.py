"""Independent verifier and analyzer for the Acrobot ProCuRL selection study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

import gymnasium
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RAW_SCHEMA = "curriculum-maxrl/acrobot-procurl-selection-raw/v1"
LOCK_SCHEMA = "curriculum-maxrl/acrobot-procurl-selection-lock/v1"
GATE_SCHEMA = "curriculum-maxrl/acrobot-procurl-selection-development-gates/v1"
ANALYSIS_SCHEMA = "curriculum-maxrl/acrobot-procurl-selection-analysis/v1"
LOCK_PATH = HERE / "ACROBOT_PROCURL_SELECTION_LOCK.json"
V2_LOCK_PATH = HERE / "ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json"
V2_DEPENDENCY_PATHS = (
    "frontier_rl/examples/run_acrobot_neural.py",
    "frontier_rl/examples/test_run_acrobot_neural.py",
    "frontier_rl/__init__.py",
    "frontier_rl/adapters/__init__.py",
    "frontier_rl/adapters/acrobot_neural.py",
    "frontier_rl/teacher.py",
    "frontier_rl/estimators.py",
    "frontier_rl/interfaces.py",
    "frontier_rl/trainer.py",
)
LOCK_KEYS = {
    "schema",
    "status",
    "created_utc",
    "purpose",
    "runtime",
    "schedule",
    "seed_collision_audit",
    "source_sha256",
    "v2_dependency_audit",
}

ARM_NAMES = (
    "procurl_env_b20_f5120",
    "probe_sham_uniform_f5120",
    "ordinary_uniform",
    "u16_probe_range_matched_f5120",
)
PROBED_ARMS = frozenset(
    {
        "procurl_env_b20_f5120",
        "probe_sham_uniform_f5120",
        "u16_probe_range_matched_f5120",
    }
)
ADAPTIVE_ARMS = frozenset({"procurl_env_b20_f5120", "u16_probe_range_matched_f5120"})
THRESHOLDS = (-1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.7, 1.0)
MIDPOINTS = tuple((left + right) / 2.0 for left, right in pairwise(THRESHOLDS))
N_ROLLOUTS = 16
PROBES_PER_TASK = 20
REFRESH_STUDENT_TRANSITIONS = 5_120
REGULAR_EVAL_INTERVAL_PAID = 100_000
MAX_EPISODE_STEPS = 500
MAX_STUDENT_GROUP_TRANSITIONS = 8_000
MAX_PROBE_SWEEP_TRANSITIONS = 80_000
CONFIRMATORY_SEEDS = tuple(range(21_000, 21_080))
DEVELOPMENT_SEEDS = tuple(range(21_300, 21_303))
QUICK_SEEDS = (21_400,)
CONFIRMATORY_PAID_BUDGET = 2_000_000
DEVELOPMENT_PAID_BUDGET = 400_000
QUICK_PAID_BUDGET = 100_000
CONFIRMATORY_EVAL_N = 32
DEVELOPMENT_EVAL_N = 32
QUICK_EVAL_N = 2
PROCURL_BETA = 20.0
U16_BETA_CONTINUOUS_RANGE_MATCHED = 6.416133525771289
U16_LATTICE_MAX_LOGIT = 4.97730861318145
PRIMARY_SESOI = 0.02
BOOTSTRAP_RESAMPLES = 20_000
SIGN_FLIP_MONTE_CARLO_DRAWS = 1_000_000
UPSTREAM_PROCURL_COMMIT = "17904f1d7b9b29e089d4f70ae7aadf1da50ba6b2"
ENGINE_MASTER_BASE = 50_000_000_000
ENGINE_MASTER_STRIDE = 10_000_000
ENVIRONMENT_ADAPTER_SEED_OFFSET = 1_000
RNG_DOMAIN_OFFSETS = {
    "actor_parameter": 0,
    "actor_action": 1,
    "selection": 10_000,
    "environment_reset_rng": 11_003,
    "evaluation_episode": 1_000_000,
    "evaluation_action": 1_000_001,
    "probe_episode_reset": 2_000_000,
    "probe_episode_action": 3_000_000,
}
MAX_ENCODED_PROBE_SWEEPS = 512
PROBE_COORDINATE_BLOCK = MAX_ENCODED_PROBE_SWEEPS * 8 * PROBES_PER_TASK
EVALUATION_RESET_NAMESPACE_BASE = 1_800_000_000
EVALUATION_ACTION_NAMESPACE_BASE = 4_000_000_000_000_000_000
PROBE_ACTION_NAMESPACE_BASE = 3_000_000_000_000_000_000
PINNED_RUNTIME_VERSIONS = {
    "python_implementation": "CPython",
    "python": "3.12.13",
    "numpy": "2.5.1",
    "gymnasium": "1.3.0",
}

DEVELOPMENT_GATE_NAMES = (
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
DEVELOPMENT_GATE_POLICY = {
    "outcome_blind": True,
    "uses_arm_contrasts": False,
    "uses_effect_direction": False,
    "uses_confidence_intervals": False,
    "uses_p_values": False,
    "uses_minimum_effect": False,
}

EXPECTED_SOURCE_RELATIVE_PATHS = (
    "frontier_rl/examples/run_acrobot_procurl_selection.py",
    "frontier_rl/examples/analyze_acrobot_procurl_selection.py",
    "frontier_rl/examples/build_acrobot_procurl_selection_lock.py",
    "frontier_rl/examples/verify_acrobot_procurl_selection_portable.py",
    "frontier_rl/examples/test_run_acrobot_procurl_selection.py",
    "frontier_rl/examples/test_analyze_acrobot_procurl_selection.py",
    "frontier_rl/examples/test_build_acrobot_procurl_selection_lock.py",
    "frontier_rl/examples/test_verify_acrobot_procurl_selection_portable.py",
    "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_PROTOCOL.md",
    "frontier_rl/examples/PROCURL_PRIMARY_SOURCE_PROVENANCE.md",
    "frontier_rl/examples/INVALID_ABORTED_PRE_GATE_ENTROPY_SUM_MISMATCH/INCIDENT.json",
    "frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json",
    "frontier_rl/examples/run_acrobot_neural.py",
    "frontier_rl/examples/test_run_acrobot_neural.py",
    "frontier_rl/__init__.py",
    "frontier_rl/adapters/__init__.py",
    "frontier_rl/adapters/acrobot_neural.py",
    "frontier_rl/teacher.py",
    "frontier_rl/estimators.py",
    "frontier_rl/interfaces.py",
    "frontier_rl/trainer.py",
)

PRIOR_LOGICAL_SEED_BLOCKS = {
    "legacy_acrobot_core": tuple(range(20)),
    "legacy_acrobot_scale": tuple(range(100, 110)),
    "acrobot_v1_pilot": tuple(range(10_000, 10_003)),
    "acrobot_v2_development": tuple(range(11_000, 11_003)),
    "acrobot_v3_confirmation": tuple(range(12_000, 12_020)),
    "acrobot_hindsight_v4a": tuple(range(13_000, 13_003)),
    "acrobot_hindsight_v4b": tuple(range(14_000, 14_010)),
    "acrobot_hindsight_v5a": tuple(range(15_000, 15_003)),
    "acrobot_hindsight_v5b": tuple(range(16_000, 16_020)),
    "mountaincar_v1_development": tuple(range(17_000, 17_003)),
    "mountaincar_v1_reserved_confirmation": tuple(range(18_000, 18_020)),
    "aborted_tournament_confirmation": tuple(range(19_000, 19_020)),
    "aborted_tournament_development": tuple(range(19_100, 19_103)),
    "acrobot_tournament_confirmation": tuple(range(20_000, 20_020)),
    "acrobot_tournament_development": tuple(range(20_100, 20_103)),
    "acrobot_tournament_quick": (20_200,),
    "invalid_procurl_selection_development_pre_gate_entropy_sum_mismatch": tuple(
        range(21_100, 21_103)
    ),
    "invalid_procurl_selection_quick_pre_gate_entropy_sum_mismatch": (21_200,),
}

SCORE_KEYS = (
    "evaluation_shared_full_horizon_trajectories",
    "episode_records",
    "max_heights",
    "pass_rates",
    "target_uniform_mean_success",
    "midpoint_thresholds",
    "midpoint_pass_rates",
    "native_success_rate",
    "mean_native_return",
    "mean_censored_time_to_goal",
    "policy_entropy_sum",
    "policy_entropy_count",
    "mean_policy_entropy",
    "actor_parameter_fingerprint",
)

RAW_KEYS = {
    "schema",
    "artifact_state",
    "provenance",
    "protocol",
    "run_failures",
    "cases",
}
PROVENANCE_KEYS = {
    "created_utc",
    "runtime",
    "source_lock_relative_path",
    "source_lock_sha256",
    "source_lock_enforced",
    "source_sha256",
    "git_commit",
    "git_status_porcelain",
    "seed_collision_audit",
    "upstream_procurl_commit",
    "upstream_code_copied",
    "public_preexecution_registration",
}
CASE_KEYS = {"config", "summary", "runs"}
CONFIG_KEYS = {"name", "selection", "probes"}
SUMMARY_KEYS = {"n_attempted", "n_valid", "n_failed", "ledger_means_descriptive_only"}
SUMMARY_LEDGER_KEYS = {
    "paid_transitions",
    "student_transitions",
    "probe_transitions",
    "probe_sweeps",
    "optimizer_updates",
}
RUN_KEYS = {
    "logical_seed",
    "engine_master_seed",
    "environment_adapter_seed_argument",
    "rng_roots",
    "seed",
    "numeric_valid",
    "paid_budget_nominal",
    "paid_transitions",
    "paid_budget_overshoot",
    "student_transitions",
    "probe_transitions",
    "probe_fraction_of_paid",
    "sampled_groups",
    "student_rollouts",
    "probe_episodes",
    "probe_sweeps",
    "optimizer_updates",
    "live_groups",
    "dead_groups",
    "all_pass_groups",
    "zero_gradient_update_attempts",
    "task_groups",
    "task_rollouts",
    "task_successes",
    "task_student_transitions",
    "total_parameters",
    "active_parameters_per_task",
    "final_parameter_fingerprint",
    "initial_parameter_fingerprint",
    "wall_seconds",
    "paid_transitions_per_wall_second",
    "group_records",
    "probe_sweep_records",
    "evaluation_records",
    "selection_diagnostics",
    "accounting_valid",
    "evaluation_rng_preserved",
    "probe_training_state_preserved",
}
SELECTION_KEYS = {
    "p_hat",
    "utility",
    "logits",
    "probabilities",
    "estimates_used_for_selection",
}
SWEEP_KEYS = {
    "sweep_ordinal",
    "trigger",
    "crossed_boundary_student_transition",
    "student_transitions",
    "sampled_groups",
    "optimizer_updates",
    "paid_before",
    "paid_after",
    "probe_transitions",
    "task_records",
    "p_hat",
    "selection_after_sweep",
    "training_state_fingerprint_before",
    "training_state_fingerprint_after",
    "parameter_fingerprint_before",
    "parameter_fingerprint_after",
    "actor_update_calls_before",
    "actor_update_calls_after",
    "actor_applied_updates_before",
    "actor_applied_updates_after",
    "training_state_preserved",
    "pre_probe_evaluation_id",
    "post_probe_copy_evaluation_id",
}
PROBE_TASK_KEYS = {
    "task_id",
    "threshold",
    "n_episodes",
    "success_count",
    "p_hat",
    "transitions",
    "episodes",
}
PROBE_EPISODE_KEYS = {
    "episode",
    "reset_seed",
    "action_seed",
    "success",
    "transitions",
    "max_height",
    "action_count",
    "action_sha256",
}
GROUP_KEYS = {
    "group",
    "task_id",
    "threshold",
    "selection_source_sweep_ordinal",
    "selection_draw_index",
    "selection_uniform",
    "selection_rng_fingerprint_before",
    "selection_rng_fingerprint_after",
    "selection_probabilities_before_group",
    "selected_task_probability",
    "p_hat_used_before_group",
    "utility_used_before_group",
    "logits_used_before_group",
    "paid_before_group",
    "paid_after_student_group",
    "paid_after_required_sweeps",
    "student_transition_start",
    "student_transition_end",
    "student_transitions",
    "student_success_flags",
    "student_rollout_records",
    "required_crossed_boundaries",
    "required_sweep_ordinals",
    "success_count",
    "regime",
    "realized_practical_maxrl_abs_coefficient_mass",
    "optimizer_updates_after_group",
    "update",
    "actor_parameter_fingerprint_before_group",
    "actor_parameter_fingerprint_after_group",
    "actor_action_rng_fingerprint_before_student",
    "actor_action_rng_fingerprint_after_student",
    "environment_reset_rng_fingerprint_before_student",
    "environment_reset_rng_fingerprint_after_student",
}
STUDENT_ROLLOUT_KEYS = {
    "rollout",
    "student_reset_draw_index",
    "reset_seed",
    "success",
    "transitions",
    "max_height",
    "action_count",
    "action_sha256",
}
UPDATE_KEYS = {
    "source",
    "eligible",
    "requested",
    "weights",
    "weight_sum",
    "weight_l1",
    "parameter_fingerprint_before",
    "parameter_fingerprint_after",
    "actor_update_calls_before",
    "actor_update_calls_after",
    "actor_applied_updates_before",
    "actor_applied_updates_after",
    "applied",
    "diagnostics",
}
UPDATE_DIAGNOSTIC_KEYS = {"gradient_norm", "update_norm", "mean_policy_entropy"}
EVALUATION_KEYS = {
    "evaluation_id",
    "kind",
    "paid_transitions",
    "student_transitions",
    "probe_transitions",
    "sampled_groups",
    "optimizer_updates",
    "sweep_count",
    "crossed_regular_paid_thresholds",
    "copied_from_evaluation_id",
    "evaluation_was_executed",
    "evaluation_shared_full_horizon_trajectories",
    "episode_records",
    "max_heights",
    "pass_rates",
    "target_uniform_mean_success",
    "midpoint_thresholds",
    "midpoint_pass_rates",
    "native_success_rate",
    "mean_native_return",
    "mean_censored_time_to_goal",
    "policy_entropy_sum",
    "policy_entropy_count",
    "mean_policy_entropy",
    "actor_parameter_fingerprint",
    "training_state_fingerprint_before",
    "training_state_fingerprint_after",
    "training_state_preserved",
}
EVALUATION_EPISODE_KEYS = {
    "episode",
    "reset_seed",
    "action_seed",
    "transitions",
    "native_success",
    "native_return",
    "censored_time_to_goal",
    "max_height",
    "policy_entropy_sum",
    "policy_entropy_count",
    "action_count",
    "action_sha256",
}
SELECTION_DIAGNOSTIC_KEYS = {
    "mean_selection_entropy",
    "mean_selection_tv_from_uniform",
    "mean_max_task_probability",
    "mean_assigned_probability_per_task",
    "realized_task_fraction",
    "realized_task_tv_from_uniform",
    "student_fraction_of_paid",
    "probe_fraction_of_paid",
    "paid_budget_overshoot",
}
GATE_KEYS = {
    "schema",
    "mode",
    "all_gates_passed",
    "source_lock_sha256",
    "source_lock_verification",
    "gates",
    "diagnostics",
    "gate_policy",
    "raw_artifact_relative_path",
    "raw_artifact_sha256",
}
BINDING_KEYS = {
    "relative_path",
    "sha256",
    "raw_artifact_relative_path",
    "raw_artifact_sha256",
    "all_gates_passed",
}
RUNTIME_KEYS = {
    "python_implementation",
    "python",
    "platform",
    "machine",
    "numpy",
    "gymnasium",
}
SOURCE_VERIFICATION_KEYS = {
    "passed",
    "runtime",
    "source_lock_sha256",
    "checked_source_files",
}

ARM_CONFIGS = {
    "procurl_env_b20_f5120": {
        "name": "procurl_env_b20_f5120",
        "selection": "procurl_p1mp_softmax",
        "probes": True,
    },
    "probe_sham_uniform_f5120": {
        "name": "probe_sham_uniform_f5120",
        "selection": "uniform_sham",
        "probes": True,
    },
    "ordinary_uniform": {
        "name": "ordinary_uniform",
        "selection": "uniform_ordinary",
        "probes": False,
    },
    "u16_probe_range_matched_f5120": {
        "name": "u16_probe_range_matched_f5120",
        "selection": "u16_softmax",
        "probes": True,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_utc_iso8601(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(
        None
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def load_strict_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} as strict JSON: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain exactly one JSON object")
    return payload


def _runtime() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
    }


def _mode_schedule(mode: str) -> tuple[tuple[int, ...], int, int]:
    if mode == "confirmatory":
        return CONFIRMATORY_SEEDS, CONFIRMATORY_PAID_BUDGET, CONFIRMATORY_EVAL_N
    if mode == "development":
        return DEVELOPMENT_SEEDS, DEVELOPMENT_PAID_BUDGET, DEVELOPMENT_EVAL_N
    if mode == "quick":
        return QUICK_SEEDS, QUICK_PAID_BUDGET, QUICK_EVAL_N
    raise ValueError(f"unknown mode {mode!r}")


def _independent_protocol(mode: str, development_gate: dict | None) -> dict:
    """Reconstruct every frozen protocol field without importing the runner."""
    seeds, paid_budget, eval_n = _mode_schedule(mode)
    return {
        "study": "acrobot_procurl_selection_semantics",
        "mode": mode,
        "status": (
            "confirmatory"
            if mode == "confirmatory"
            else "engineering_only_no_scientific_inference"
            if mode == "quick"
            else "development_only"
        ),
        "protocol_document": (
            "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_PROTOCOL.md"
        ),
        "primary_source_provenance": (
            "frontier_rl/examples/PROCURL_PRIMARY_SOURCE_PROVENANCE.md"
        ),
        "upstream_procurl_commit": UPSTREAM_PROCURL_COMMIT,
        "arm_names": list(ARM_NAMES),
        "paired_logical_seeds": list(seeds),
        "logical_to_engine_master_seed": {
            str(seed): _engine_master_seed(seed) for seed in seeds
        },
        "paid_budget_nominal": paid_budget,
        "complete_atomic_overshoot_retained": True,
        "paid_definition": (
            "student transitions plus probe transitions; evaluation excluded"
        ),
        "student_group_size": N_ROLLOUTS,
        "student_estimator": "practical N=16 dropped-group MaxRL",
        "architecture": "task-blind shared H64, 640 parameters",
        "learning_rate": 3e-4,
        "optimizer": "plain SGD ascent",
        "hindsight": False,
        "thresholds": list(THRESHOLDS),
        "probe": {
            "episodes_per_task_per_sweep": PROBES_PER_TASK,
            "tasks": len(THRESHOLDS),
            "initial_sweep": True,
            "refresh_clock": "student transitions",
            "refresh_every": REFRESH_STUDENT_TRANSITIONS,
            "boundary_mapping": (
                "after complete student group and before its update, run one complete "
                "sweep per crossed boundary"
            ),
            "paid": True,
            "actor_and_training_rng_nonmutating": True,
            "latest_complete_sweep_replaces_previous_estimates": True,
        },
        "selection_rules": {
            "procurl_env_b20_f5120": "softmax(20*p_hat*(1-p_hat))",
            "probe_sham_uniform_f5120": "exact 1/8 after discarded probes",
            "ordinary_uniform": "exact 1/8 with no probes",
            "u16_probe_range_matched_f5120": (
                f"softmax({U16_BETA_CONTINUOUS_RANGE_MATCHED}*"
                "(1-(1-p_hat)^16-p_hat)); continuous-domain max logit 5; "
                f"0.05-lattice max logit {U16_LATTICE_MAX_LOGIT}"
            ),
        },
        "evaluation": {
            "shared_full_horizon_trajectories": eval_n,
            "common_episode_reset_and_action_streams": True,
            "primary_score": "unweighted mean success over eight thresholds",
            "raw_max_heights_retained": True,
            "seven_midpoint_thresholds_retained": True,
            "regular_interval_paid": REGULAR_EVAL_INTERVAL_PAID,
            "initial_terminal_and_around_every_sweep": True,
            "post_probe_score_is_pre_probe_copy": True,
        },
        "primary": {
            "contrast": ("u16_probe_range_matched_f5120 minus procurl_env_b20_f5120"),
            "metric": ("fixed-nominal-paid-budget target-uniform mean-success AUC"),
            "test": "two-sided paired t-test",
            "alpha": 0.05,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": 31_000,
            "bootstrap_quantiles": [0.025, 0.975],
            "bootstrap_quantile_method": "linear",
            "sesoi": PRIMARY_SESOI,
            "decision": "mean >= +0.02 and p <= 0.05",
            "robustness": (
                f"deterministic Monte Carlo paired sign flip, "
                f"{SIGN_FLIP_MONTE_CARLO_DRAWS} draws, seed 31001, "
                "plus-one correction"
            ),
        },
        "secondary_holm_family": [
            "procurl-minus-sham",
            "u16-minus-sham",
            "procurl-minus-ordinary",
            "u16-minus-ordinary",
            "sham-minus-ordinary",
        ],
        "secondary_statistics": {
            "paired_t_alpha": 0.05,
            "holm_familywise_alpha": 0.05,
            "bootstrap_seeds_in_family_order": list(range(31_100, 31_105)),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_quantiles": [0.025, 0.975],
            "bootstrap_quantile_method": "linear",
        },
        "auc_convention": {
            "integration": "trapezoid on nondecreasing recorded coordinates",
            "duplicate_coordinates": (
                "retain in record order; zero width contributes zero area; "
                "last duplicate governs the next positive-width segment"
            ),
            "nominal_cutoff": "linear interpolation within the crossing segment",
            "normalization": "divide by cutoff minus zero",
        },
        "development_gate": development_gate,
        "rng_domain_contract": {
            "engine_master_base": ENGINE_MASTER_BASE,
            "engine_master_stride": ENGINE_MASTER_STRIDE,
            "rng_domain_offsets": dict(RNG_DOMAIN_OFFSETS),
            "environment_adapter_seed_offset": ENVIRONMENT_ADAPTER_SEED_OFFSET,
            "pairing": "all four arms share roots within a logical seed",
        },
        "no_public_preexecution_commit": True,
        "claim_scope": (
            "source-faithful ProCuRL selection semantics attached to this learner; "
            "not full PPO ProCuRL reproduction"
        ),
        "raw_only": True,
    }


def _independent_locked_schedule() -> dict:
    return {
        "arm_names": list(ARM_NAMES),
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "quick_seeds": list(QUICK_SEEDS),
        "confirmatory_paid_budget": CONFIRMATORY_PAID_BUDGET,
        "development_paid_budget": DEVELOPMENT_PAID_BUDGET,
        "quick_paid_budget": QUICK_PAID_BUDGET,
        "regular_eval_interval_paid": REGULAR_EVAL_INTERVAL_PAID,
        "confirmatory_eval_n": CONFIRMATORY_EVAL_N,
        "development_eval_n": DEVELOPMENT_EVAL_N,
        "quick_eval_n": QUICK_EVAL_N,
        "n_rollouts": N_ROLLOUTS,
        "learning_rate": 3e-4,
        "probes_per_task": PROBES_PER_TASK,
        "refresh_student_transitions": REFRESH_STUDENT_TRANSITIONS,
        "procurl_beta": PROCURL_BETA,
        "u16_beta_continuous_range_matched": U16_BETA_CONTINUOUS_RANGE_MATCHED,
        "u16_lattice_max_logit": U16_LATTICE_MAX_LOGIT,
        "engine_master_base": ENGINE_MASTER_BASE,
        "engine_master_stride": ENGINE_MASTER_STRIDE,
        "rng_domain_offsets": dict(RNG_DOMAIN_OFFSETS),
        "environment_adapter_seed_offset": ENVIRONMENT_ADAPTER_SEED_OFFSET,
        "upstream_procurl_commit": UPSTREAM_PROCURL_COMMIT,
    }


def _engine_master_seed(logical_seed: int) -> int:
    return ENGINE_MASTER_BASE + logical_seed * ENGINE_MASTER_STRIDE


def _rng_domain_record(logical_seed: int) -> dict:
    master = _engine_master_seed(logical_seed)
    return {
        "logical_seed": logical_seed,
        "engine_master_seed": master,
        "environment_adapter_seed_argument": master + ENVIRONMENT_ADAPTER_SEED_OFFSET,
        "rng_roots": {key: master + value for key, value in RNG_DOMAIN_OFFSETS.items()},
    }


def _independent_seed_collision_audit() -> dict:
    registered = CONFIRMATORY_SEEDS + DEVELOPMENT_SEEDS + QUICK_SEEDS
    prior = set().union(*(set(values) for values in PRIOR_LOGICAL_SEED_BLOCKS.values()))
    blocks = {
        "confirmatory_vs_prior": sorted(set(CONFIRMATORY_SEEDS) & prior),
        "development_vs_prior": sorted(set(DEVELOPMENT_SEEDS) & prior),
        "quick_vs_prior": sorted(set(QUICK_SEEDS) & prior),
        "confirmatory_vs_development": sorted(
            set(CONFIRMATORY_SEEDS) & set(DEVELOPMENT_SEEDS)
        ),
        "confirmatory_vs_quick": sorted(set(CONFIRMATORY_SEEDS) & set(QUICK_SEEDS)),
        "development_vs_quick": sorted(set(DEVELOPMENT_SEEDS) & set(QUICK_SEEDS)),
    }
    records = {str(seed): _rng_domain_record(seed) for seed in registered}
    owners: dict[int, tuple[int, str]] = {}
    derived_collisions = []
    for seed in registered:
        for domain, root in records[str(seed)]["rng_roots"].items():
            if root in owners:
                derived_collisions.append(
                    {
                        "root": root,
                        "first": list(owners[root]),
                        "second": [seed, domain],
                    }
                )
            owners[root] = (seed, domain)
    derived_vs_logical = sorted(set(owners) & (prior | set(registered)))
    expected = len(registered) * len(RNG_DOMAIN_OFFSETS)
    passed = (
        not any(blocks.values())
        and not derived_collisions
        and not derived_vs_logical
        and len(owners) == expected
        and len(CONFIRMATORY_SEEDS) == 80
        and len(DEVELOPMENT_SEEDS) == 3
        and len(QUICK_SEEDS) == 1
    )
    return {
        "passed": passed,
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "quick_seeds": list(QUICK_SEEDS),
        "prior_logical_seed_blocks": {
            key: list(value) for key, value in PRIOR_LOGICAL_SEED_BLOCKS.items()
        },
        "logical_collisions": blocks,
        "derived_root_collisions": derived_collisions,
        "derived_roots_vs_logical_seeds": derived_vs_logical,
        "unique_derived_root_count": len(owners),
        "expected_unique_derived_root_count": expected,
        "per_logical_seed": records,
        "pairing": "all four arms deliberately reuse roots within logical seed",
        "collision_free_episode_seed_encoding": {
            "passed": True,
            "probe_reset_formula": (
                "((logical_seed*512+sweep_ordinal-1)*8+task_id)*20+episode"
            ),
            "probe_sweep_slots_per_seed": MAX_ENCODED_PROBE_SWEEPS,
            "probe_coordinate_block_per_seed": PROBE_COORDINATE_BLOCK,
            "maximum_registered_probe_reset_seed": (
                ((max(registered) * MAX_ENCODED_PROBE_SWEEPS + 511) * 8 + 7)
                * PROBES_PER_TASK
                + 19
            ),
            "evaluation_reset_namespace_base": EVALUATION_RESET_NAMESPACE_BASE,
            "probe_action_namespace_base": PROBE_ACTION_NAMESPACE_BASE,
            "evaluation_action_namespace_base": EVALUATION_ACTION_NAMESPACE_BASE,
            "coordinate_ranges_disjoint": True,
        },
    }


def _probe_episode_seeds(
    logical_seed: int, sweep_ordinal: int, task_id: int, episode: int
) -> tuple[int, int]:
    _require(
        logical_seed in CONFIRMATORY_SEEDS + DEVELOPMENT_SEEDS + QUICK_SEEDS,
        "probe logical seed is not registered",
    )
    _require(
        type(sweep_ordinal) is int and 1 <= sweep_ordinal <= MAX_ENCODED_PROBE_SWEEPS,
        "probe sweep ordinal exceeds collision-free namespace",
    )
    _require(
        type(task_id) is int and 0 <= task_id < 8,
        "probe task coordinate invalid",
    )
    _require(
        type(episode) is int and 0 <= episode < PROBES_PER_TASK,
        "probe episode coordinate invalid",
    )
    coordinate = (
        (logical_seed * MAX_ENCODED_PROBE_SWEEPS + (sweep_ordinal - 1)) * 8 + task_id
    ) * PROBES_PER_TASK + episode
    return coordinate, PROBE_ACTION_NAMESPACE_BASE + coordinate


def _evaluation_episode_seeds(logical_seed: int, episode: int) -> tuple[int, int]:
    _require(
        logical_seed in CONFIRMATORY_SEEDS + DEVELOPMENT_SEEDS + QUICK_SEEDS,
        "evaluation logical seed is not registered",
    )
    _require(
        type(episode) is int and 0 <= episode < CONFIRMATORY_EVAL_N,
        "evaluation episode coordinate invalid",
    )
    coordinate = (
        logical_seed - min(CONFIRMATORY_SEEDS)
    ) * CONFIRMATORY_EVAL_N + episode
    return (
        EVALUATION_RESET_NAMESPACE_BASE + coordinate,
        EVALUATION_ACTION_NAMESPACE_BASE + coordinate,
    )


def verify_source_lock(raw: dict, lock: dict, lock_path: Path) -> dict:
    errors = []
    runtime = _runtime()
    if lock_path.resolve() != LOCK_PATH.resolve():
        errors.append("noncanonical lock path")
    if lock.get("schema") != LOCK_SCHEMA:
        errors.append("lock schema mismatch")
    if set(lock) != LOCK_KEYS:
        errors.append("lock top-level schema is not closed")
    if lock.get("status") != "sealed_before_any_quick_development_or_confirmation":
        errors.append("lock status mismatch")
    if lock.get("purpose") != (
        "Canonical pre-execution source/runtime lock for the Acrobot "
        "ProCuRL selection-semantic study."
    ):
        errors.append("lock purpose mismatch")
    if not _is_utc_iso8601(lock.get("created_utc")):
        errors.append("lock creation timestamp invalid")
    if {
        key: runtime[key] for key in PINNED_RUNTIME_VERSIONS
    } != PINNED_RUNTIME_VERSIONS:
        errors.append("runtime versions are not pinned")
    if lock.get("runtime") != runtime:
        errors.append("lock runtime mismatch")
    if lock.get("schedule") != _independent_locked_schedule():
        errors.append("lock schedule mismatch")
    audit = _independent_seed_collision_audit()
    if lock.get("seed_collision_audit") != audit or not audit["passed"]:
        errors.append("lock seed audit mismatch")
    try:
        v2 = load_strict_json(V2_LOCK_PATH, "V2 Acrobot dependency lock")
        v2_hashes = v2.get("source_sha256", {})
        live_dependencies = {
            relative: _sha256(PROJECT_ROOT / relative)
            for relative in V2_DEPENDENCY_PATHS
        }
        frozen_dependencies = {
            relative: v2_hashes[relative] for relative in V2_DEPENDENCY_PATHS
        }
        expected_v2_audit = {
            "passed": True,
            "v2_lock_relative_path": str(V2_LOCK_PATH.relative_to(PROJECT_ROOT)),
            "v2_lock_sha256": _sha256(V2_LOCK_PATH),
            "v2_lock_schema": v2["schema"],
            "dependency_paths": list(V2_DEPENDENCY_PATHS),
            "live_dependency_sha256": live_dependencies,
            "v2_locked_dependency_sha256": frozen_dependencies,
            "all_live_dependencies_match_v2": True,
        }
        if (
            v2.get("schema") != "curriculum-maxrl/acrobot-curriculum-tournament-lock/v2"
            or live_dependencies != frozen_dependencies
            or lock.get("v2_dependency_audit") != expected_v2_audit
        ):
            errors.append("V2 dependency audit mismatch")
    except (KeyError, ValueError, OSError):
        errors.append("V2 dependency audit could not be reconstructed")
    live_hashes = {
        relative: _sha256(PROJECT_ROOT / relative)
        for relative in EXPECTED_SOURCE_RELATIVE_PATHS
        if (PROJECT_ROOT / relative).is_file()
    }
    if set(live_hashes) != set(EXPECTED_SOURCE_RELATIVE_PATHS):
        errors.append("locked source file missing")
    if set(lock.get("source_sha256", {})) != set(EXPECTED_SOURCE_RELATIVE_PATHS):
        errors.append("source key set mismatch")
    if lock.get("source_sha256") != live_hashes:
        errors.append("source hashes mismatch")
    lock_hash = _sha256(lock_path)
    provenance = raw.get("provenance", {})
    if provenance.get("source_lock_sha256") != lock_hash:
        errors.append("raw lock hash mismatch")
    if provenance.get("source_lock_enforced") is not True:
        errors.append("raw did not enforce lock")
    if provenance.get("source_sha256") != live_hashes:
        errors.append("raw source hashes mismatch")
    if provenance.get("runtime") != runtime:
        errors.append("raw runtime mismatch")
    if provenance.get("seed_collision_audit") != audit:
        errors.append("raw seed audit mismatch")
    if errors:
        raise ValueError("source/runtime verification failed: " + "; ".join(errors))
    return {
        "passed": True,
        "runtime": runtime,
        "source_lock_sha256": lock_hash,
        "checked_source_files": sorted(live_hashes),
    }


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - float(np.max(logits))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def _recompute_selection(arm_name: str, p_hat: Sequence[float] | None) -> dict:
    uniform = np.full(8, 1.0 / 8.0)
    if arm_name in {"probe_sham_uniform_f5120", "ordinary_uniform"}:
        return {
            "p_hat": None if p_hat is None else [float(x) for x in p_hat],
            "utility": None,
            "logits": None,
            "probabilities": uniform.tolist(),
            "estimates_used_for_selection": False,
        }
    if p_hat is None:
        raise ValueError("adaptive selection lacks p_hat")
    p = np.asarray(p_hat, dtype=np.float64)
    if arm_name == "procurl_env_b20_f5120":
        utility = p * (1.0 - p)
        logits = PROCURL_BETA * utility
    elif arm_name == "u16_probe_range_matched_f5120":
        utility = 1.0 - np.power(1.0 - p, 16) - p
        logits = U16_BETA_CONTINUOUS_RANGE_MATCHED * utility
    else:
        raise ValueError(f"unknown arm {arm_name!r}")
    return {
        "p_hat": p.tolist(),
        "utility": utility.tolist(),
        "logits": logits.tolist(),
        "probabilities": _stable_softmax(logits).tolist(),
        "estimates_used_for_selection": True,
    }


def _crossed_boundaries(before: int, after: int) -> list[int]:
    return [
        index * REFRESH_STUDENT_TRANSITIONS
        for index in range(
            before // REFRESH_STUDENT_TRANSITIONS + 1,
            after // REFRESH_STUDENT_TRANSITIONS + 1,
        )
    ]


def _close(left, right, *, atol: float = 1e-12) -> bool:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        return False
    return bool(
        np.allclose(left_array, right_array, rtol=0.0, atol=atol, equal_nan=False)
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_exact_keys(value: object, expected: set[str], label: str) -> dict:
    _require(isinstance(value, dict), f"{label} must be an object")
    observed = set(value)
    _require(
        observed == expected,
        f"{label} schema mismatch: missing={sorted(expected - observed)}, "
        f"extra={sorted(observed - expected)}",
    )
    return value


def _require_sha256(value: object, label: str) -> None:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256",
    )


def _require_vector_shape(value: object, length: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _require(array.shape == (length,), f"{label} must have shape ({length},)")
    _require(np.isfinite(array).all(), f"{label} contains nonfinite values")
    return array


def _rng_state_fingerprint(rng: np.random.Generator) -> str:
    return hashlib.sha256(pickle.dumps(rng.bit_generator.state, protocol=5)).hexdigest()


def _validate_sweeps(arm_name: str, run: dict) -> dict:
    sweeps = run.get("probe_sweep_records")
    _require(isinstance(sweeps, list), "probe_sweep_records must be a list")
    probed = arm_name in PROBED_ARMS
    expected_count = (
        1 + int(run["student_transitions"]) // REFRESH_STUDENT_TRANSITIONS
        if probed
        else 0
    )
    _require(len(sweeps) == expected_count, "probe sweep count/cadence mismatch")
    if not probed:
        _require(sweeps == [] and run["probe_transitions"] == 0, "ordinary arm probed")
        return {
            "all_exact_probe_count_and_bounded": True,
            "all_p_hat_multiple": True,
            "schedule_exact": True,
            "state_preserved": True,
            "adaptive_nonuniform_seen": False,
        }

    expected_boundaries = list(
        range(
            REFRESH_STUDENT_TRANSITIONS,
            (int(run["student_transitions"]) // REFRESH_STUDENT_TRANSITIONS)
            * REFRESH_STUDENT_TRANSITIONS
            + 1,
            REFRESH_STUDENT_TRANSITIONS,
        )
    )
    observed_boundaries = []
    adaptive_nonuniform = False
    summed_probe_transitions = 0
    for index, sweep in enumerate(sweeps, start=1):
        _require_exact_keys(sweep, SWEEP_KEYS, f"probe sweep {index}")
        _require(
            sweep.get("sweep_ordinal") == index, "sweep ordinals are not consecutive"
        )
        if index == 1:
            _require(
                sweep.get("trigger") == "initial"
                and sweep.get("crossed_boundary_student_transition") is None
                and sweep.get("student_transitions") == 0
                and sweep.get("sampled_groups") == 0
                and sweep.get("optimizer_updates") == 0
                and sweep.get("paid_before") == 0,
                "initial sweep contract mismatch",
            )
        else:
            _require(
                sweep.get("trigger") == "refresh", "noninitial sweep trigger mismatch"
            )
            observed_boundaries.append(sweep.get("crossed_boundary_student_transition"))
        task_records = sweep.get("task_records")
        _require(
            isinstance(task_records, list) and len(task_records) == 8,
            "sweep lacks 8 tasks",
        )
        success_counts = []
        task_transition_sum = 0
        for task_id, task in enumerate(task_records):
            _require_exact_keys(
                task, PROBE_TASK_KEYS, f"probe sweep {index} task {task_id}"
            )
            _require(
                task.get("task_id") == task_id
                and task.get("threshold") == THRESHOLDS[task_id]
                and task.get("n_episodes") == PROBES_PER_TASK,
                "probe task identity/count mismatch",
            )
            episodes = task.get("episodes")
            _require(
                isinstance(episodes, list) and len(episodes) == PROBES_PER_TASK,
                "probe episode ledger length mismatch",
            )
            successes, transitions = [], []
            for episode_index, episode in enumerate(episodes):
                _require_exact_keys(
                    episode,
                    PROBE_EPISODE_KEYS,
                    f"probe sweep {index} task {task_id} episode {episode_index}",
                )
                expected_reset, expected_action = _probe_episode_seeds(
                    run["seed"], index, task_id, episode_index
                )
                _require(
                    episode.get("episode") == episode_index
                    and episode.get("reset_seed") == expected_reset
                    and episode.get("action_seed") == expected_action,
                    "probe episode coordinate/seed mismatch",
                )
                _require(
                    type(episode.get("success")) is bool, "probe success type invalid"
                )
                transition_count = episode.get("transitions")
                _require(
                    type(transition_count) is int
                    and 1 <= transition_count <= MAX_EPISODE_STEPS
                    and episode.get("action_count") == transition_count,
                    "probe episode transition/action count invalid",
                )
                _require_sha256(episode.get("action_sha256"), "probe action digest")
                max_height = episode.get("max_height")
                _require(
                    isinstance(max_height, (int, float))
                    and not isinstance(max_height, bool)
                    and math.isfinite(max_height)
                    and -2.0 <= max_height <= 2.0,
                    "probe max height invalid",
                )
                _require(
                    episode["success"] is bool(max_height > THRESHOLDS[task_id]),
                    "probe success does not equal strict threshold predicate",
                )
                successes.append(episode["success"])
                transitions.append(transition_count)
            success_count = int(sum(successes))
            transition_count = int(sum(transitions))
            _require(
                task.get("success_count") == success_count, "probe success sum mismatch"
            )
            _require(
                task.get("transitions") == transition_count,
                "probe transition sum mismatch",
            )
            _require(
                task.get("p_hat") == success_count / PROBES_PER_TASK,
                "probe p_hat mismatch",
            )
            _require(
                math.isclose(
                    task["p_hat"] * PROBES_PER_TASK, success_count, abs_tol=1e-12
                ),
                "p_hat is not an integer multiple of 0.05",
            )
            success_counts.append(success_count)
            task_transition_sum += transition_count
        probe_transitions = int(sweep.get("probe_transitions", -1))
        _require(
            1 <= probe_transitions <= MAX_PROBE_SWEEP_TRANSITIONS
            and probe_transitions == task_transition_sum,
            "probe sweep transition accounting/bound failed",
        )
        _require(
            sweep.get("paid_after") - sweep.get("paid_before") == probe_transitions,
            "probe sweep paid span mismatch",
        )
        p_hat = [count / PROBES_PER_TASK for count in success_counts]
        _require(_close(sweep.get("p_hat"), p_hat), "sweep p_hat vector mismatch")
        recomputed = _recompute_selection(arm_name, p_hat)
        observed_selection = sweep.get("selection_after_sweep")
        _require_exact_keys(observed_selection, SELECTION_KEYS, "sweep selection")
        for key in ("p_hat", "utility", "logits", "probabilities"):
            left, right = observed_selection.get(key), recomputed.get(key)
            if left is None or right is None:
                _require(left is right, f"selection {key} None mismatch")
            else:
                _require(_close(left, right), f"selection {key} recomputation mismatch")
        _require(
            observed_selection.get("estimates_used_for_selection")
            is recomputed["estimates_used_for_selection"],
            "selection estimate-use flag mismatch",
        )
        probabilities = np.asarray(recomputed["probabilities"], dtype=float)
        adaptive_nonuniform |= bool(
            arm_name in ADAPTIVE_ARMS
            and not np.array_equal(probabilities, np.full(8, 1.0 / 8.0))
        )
        for hash_key in (
            "training_state_fingerprint_before",
            "training_state_fingerprint_after",
            "parameter_fingerprint_before",
            "parameter_fingerprint_after",
        ):
            _require_sha256(sweep.get(hash_key), f"sweep {hash_key}")
        _require(
            sweep.get("training_state_preserved") is True
            and sweep.get("training_state_fingerprint_before")
            == sweep.get("training_state_fingerprint_after")
            and sweep.get("parameter_fingerprint_before")
            == sweep.get("parameter_fingerprint_after")
            and sweep.get("actor_update_calls_before")
            == sweep.get("actor_update_calls_after")
            and sweep.get("actor_applied_updates_before")
            == sweep.get("actor_applied_updates_after"),
            "probe mutated forbidden training state",
        )
        summed_probe_transitions += probe_transitions
    _require(
        observed_boundaries == expected_boundaries, "refresh boundary sequence mismatch"
    )
    _require(
        summed_probe_transitions == run["probe_transitions"], "run probe sum mismatch"
    )
    return {
        "all_exact_probe_count_and_bounded": True,
        "all_p_hat_multiple": True,
        "schedule_exact": True,
        "state_preserved": True,
        "adaptive_nonuniform_seen": adaptive_nonuniform,
    }


def _validate_evaluations(arm_name: str, run: dict, eval_n: int) -> dict:
    records = run.get("evaluation_records")
    _require(
        isinstance(records, list) and len(records) >= 2, "evaluation ledger missing"
    )
    _require(
        [record.get("evaluation_id") for record in records]
        == list(range(len(records))),
        "evaluation ids mismatch",
    )
    paid_axis = [record.get("paid_transitions") for record in records]
    _require(paid_axis == sorted(paid_axis), "evaluation paid axis is not monotone")

    executed_by_actor: dict[tuple[int, str], tuple[list[dict], dict]] = {}
    score_without_actor = tuple(
        key for key in SCORE_KEYS if key != "actor_parameter_fingerprint"
    )
    for evaluation_id, record in enumerate(records):
        _require_exact_keys(record, EVALUATION_KEYS, f"evaluation {evaluation_id}")
        _require(
            record["evaluation_id"] == evaluation_id,
            "evaluation ids are not consecutive",
        )
        for counter in (
            "paid_transitions",
            "student_transitions",
            "probe_transitions",
            "sampled_groups",
            "optimizer_updates",
            "sweep_count",
        ):
            _require(
                type(record[counter]) is int and record[counter] >= 0,
                f"evaluation {counter} invalid",
            )
        _require(
            record["paid_transitions"]
            == record["student_transitions"] + record["probe_transitions"],
            "evaluation paid ledger mismatch",
        )
        _require(
            isinstance(record["crossed_regular_paid_thresholds"], list)
            and all(
                type(value) is int and value > 0
                for value in record["crossed_regular_paid_thresholds"]
            ),
            "evaluation regular-threshold list invalid",
        )
        _require_sha256(
            record["actor_parameter_fingerprint"],
            "evaluation actor parameter fingerprint",
        )
        _require_sha256(
            record["training_state_fingerprint_before"],
            "evaluation training state before",
        )
        _require_sha256(
            record["training_state_fingerprint_after"],
            "evaluation training state after",
        )

        episode_records = record["episode_records"]
        _require(
            isinstance(episode_records, list) and len(episode_records) == eval_n,
            "evaluation episode ledger length mismatch",
        )
        maxima: list[float] = []
        native_successes: list[bool] = []
        native_returns: list[float] = []
        censored_times: list[float] = []
        entropy_terms: list[float] = []
        entropy_count = 0
        for episode_index, episode in enumerate(episode_records):
            _require_exact_keys(
                episode,
                EVALUATION_EPISODE_KEYS,
                f"evaluation {evaluation_id} episode {episode_index}",
            )
            expected_reset, expected_action = _evaluation_episode_seeds(
                run["seed"], episode_index
            )
            _require(
                episode["episode"] == episode_index
                and episode["reset_seed"] == expected_reset
                and episode["action_seed"] == expected_action,
                "evaluation episode coordinate/seed mismatch",
            )
            transitions = episode["transitions"]
            _require(
                type(transitions) is int
                and 1 <= transitions <= MAX_EPISODE_STEPS
                and episode["action_count"] == transitions,
                "evaluation episode transition/action count invalid",
            )
            _require_sha256(episode["action_sha256"], "evaluation action digest")
            max_height = episode["max_height"]
            _require(
                isinstance(max_height, (int, float))
                and not isinstance(max_height, bool)
                and math.isfinite(max_height)
                and -2.0 <= max_height <= 2.0,
                "evaluation max height invalid",
            )
            native_success = bool(max_height > 1.0)
            _require(
                type(episode["native_success"]) is bool
                and episode["native_success"] is native_success,
                "evaluation native success predicate mismatch",
            )
            _require(
                isinstance(episode["native_return"], (int, float))
                and not isinstance(episode["native_return"], bool)
                and math.isclose(
                    float(episode["native_return"]),
                    -float(transitions - int(native_success)),
                    rel_tol=0.0,
                    abs_tol=0.0,
                ),
                "evaluation native return mismatch",
            )
            expected_censored = float(
                transitions if native_success else MAX_EPISODE_STEPS
            )
            _require(
                isinstance(episode["censored_time_to_goal"], (int, float))
                and not isinstance(episode["censored_time_to_goal"], bool)
                and float(episode["censored_time_to_goal"]) == expected_censored,
                "evaluation censored time mismatch",
            )
            _require(
                type(episode["policy_entropy_count"]) is int
                and episode["policy_entropy_count"] == transitions
                and isinstance(episode["policy_entropy_sum"], (int, float))
                and not isinstance(episode["policy_entropy_sum"], bool)
                and math.isfinite(float(episode["policy_entropy_sum"]))
                and float(episode["policy_entropy_sum"]) >= 0.0
                and float(episode["policy_entropy_sum"])
                <= transitions * math.log(3.0) + 1e-12,
                "evaluation entropy ledger invalid",
            )
            maxima.append(float(max_height))
            native_successes.append(native_success)
            native_returns.append(float(episode["native_return"]))
            censored_times.append(float(episode["censored_time_to_goal"]))
            entropy_terms.append(float(episode["policy_entropy_sum"]))
            entropy_count += int(episode["policy_entropy_count"])

        # Python 3.12's built-in sum(list) uses compensated float summation.
        # Replay the runner's aggregation algorithm exactly and in ledger order.
        entropy_sum = sum(entropy_terms)

        _require(
            record["evaluation_shared_full_horizon_trajectories"] == eval_n,
            "evaluation trajectory count mismatch",
        )
        _require(_close(record["max_heights"], maxima), "evaluation maxima mismatch")
        maxima_array = np.asarray(maxima, dtype=np.float64)
        pass_rates = [
            float(np.mean(maxima_array > threshold)) for threshold in THRESHOLDS
        ]
        midpoint_rates = [
            float(np.mean(maxima_array > threshold)) for threshold in MIDPOINTS
        ]
        _require(
            _close(record["pass_rates"], pass_rates), "evaluation pass rates mismatch"
        )
        _require(
            math.isclose(
                record["target_uniform_mean_success"],
                float(np.mean(pass_rates)),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "evaluation target-uniform mean mismatch",
        )
        _require(_close(record["midpoint_thresholds"], MIDPOINTS), "midpoints mismatch")
        _require(
            _close(record["midpoint_pass_rates"], midpoint_rates),
            "midpoint rates mismatch",
        )
        summary_expectations = {
            "native_success_rate": float(np.mean(native_successes)),
            "mean_native_return": float(np.mean(native_returns)),
            "mean_censored_time_to_goal": float(np.mean(censored_times)),
            "policy_entropy_sum": entropy_sum,
            "policy_entropy_count": entropy_count,
            "mean_policy_entropy": entropy_sum / entropy_count,
        }
        for key, expected in summary_expectations.items():
            observed = record[key]
            if key == "policy_entropy_count":
                _require(observed == expected, f"evaluation {key} mismatch")
            else:
                _require(
                    isinstance(observed, (int, float))
                    and not isinstance(observed, bool)
                    and math.isclose(
                        float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
                    ),
                    f"evaluation {key} mismatch",
                )

        if record["kind"] == "post_probe_copy":
            source_id = record["copied_from_evaluation_id"]
            _require(
                type(source_id) is int and 0 <= source_id < evaluation_id,
                "copy source id invalid",
            )
            _require(
                record["evaluation_was_executed"] is False,
                "post-probe record was not marked copy",
            )
            source = records[source_id]
            for key in SCORE_KEYS:
                _require(
                    record[key] == source[key],
                    f"copied score changed for {key}",
                )
            _require(
                record["training_state_fingerprint_before"]
                == source["training_state_fingerprint_before"]
                and record["training_state_fingerprint_after"]
                == source["training_state_fingerprint_after"]
                and record["training_state_preserved"]
                is source["training_state_preserved"]
                is True,
                "post-probe copy state fields differ from source",
            )
        else:
            _require(
                record["kind"]
                in {"initial", "pre_probe", "regular_after_update", "terminal"},
                "unknown executed evaluation kind",
            )
            _require(
                record["evaluation_was_executed"] is True
                and record["copied_from_evaluation_id"] is None
                and record["training_state_preserved"] is True
                and record["training_state_fingerprint_before"]
                == record["training_state_fingerprint_after"],
                "executed evaluation mutated training state",
            )
            actor_key = (run["seed"], record["actor_parameter_fingerprint"])
            score_snapshot = {key: record[key] for key in score_without_actor}
            if actor_key in executed_by_actor:
                old_episodes, old_score = executed_by_actor[actor_key]
                _require(
                    episode_records == old_episodes and score_snapshot == old_score,
                    "same actor fingerprint produced different evaluation outputs",
                )
            else:
                executed_by_actor[actor_key] = (episode_records, score_snapshot)

    expected_events: list[dict] = [
        {
            "kind": "initial",
            "paid_transitions": 0,
            "student_transitions": 0,
            "probe_transitions": 0,
            "sampled_groups": 0,
            "optimizer_updates": 0,
            "sweep_count": 0,
            "crossed_regular_paid_thresholds": [],
            "copied_from_evaluation_id": None,
            "actor_parameter_fingerprint": run["initial_parameter_fingerprint"],
        }
    ]
    sweeps = run["probe_sweep_records"]
    if arm_name in PROBED_ARMS:
        initial_sweep = sweeps[0]
        expected_events.append(
            {
                "kind": "post_probe_copy",
                "paid_transitions": initial_sweep["paid_after"],
                "student_transitions": 0,
                "probe_transitions": initial_sweep["paid_after"],
                "sampled_groups": 0,
                "optimizer_updates": 0,
                "sweep_count": 1,
                "crossed_regular_paid_thresholds": [],
                "copied_from_evaluation_id": 0,
                "actor_parameter_fingerprint": run["initial_parameter_fingerprint"],
            }
        )

    for group in run["group_records"]:
        for ordinal in group["required_sweep_ordinals"]:
            sweep = sweeps[ordinal - 1]
            pre_id = len(expected_events)
            expected_events.append(
                {
                    "kind": "pre_probe",
                    "paid_transitions": sweep["paid_before"],
                    "student_transitions": group["student_transition_end"],
                    "probe_transitions": (
                        sweep["paid_before"] - group["student_transition_end"]
                    ),
                    "sampled_groups": group["group"],
                    "optimizer_updates": sweep["optimizer_updates"],
                    "sweep_count": ordinal - 1,
                    "crossed_regular_paid_thresholds": [],
                    "copied_from_evaluation_id": None,
                    "actor_parameter_fingerprint": group[
                        "actor_parameter_fingerprint_before_group"
                    ],
                }
            )
            expected_events.append(
                {
                    "kind": "post_probe_copy",
                    "paid_transitions": sweep["paid_after"],
                    "student_transitions": group["student_transition_end"],
                    "probe_transitions": (
                        sweep["paid_after"] - group["student_transition_end"]
                    ),
                    "sampled_groups": group["group"],
                    "optimizer_updates": sweep["optimizer_updates"],
                    "sweep_count": ordinal,
                    "crossed_regular_paid_thresholds": [],
                    "copied_from_evaluation_id": pre_id,
                    "actor_parameter_fingerprint": group[
                        "actor_parameter_fingerprint_before_group"
                    ],
                }
            )
        crossed = list(
            range(
                (group["paid_before_group"] // REGULAR_EVAL_INTERVAL_PAID + 1)
                * REGULAR_EVAL_INTERVAL_PAID,
                group["paid_after_required_sweeps"] + 1,
                REGULAR_EVAL_INTERVAL_PAID,
            )
        )
        if crossed:
            expected_events.append(
                {
                    "kind": "regular_after_update",
                    "paid_transitions": group["paid_after_required_sweeps"],
                    "student_transitions": group["student_transition_end"],
                    "probe_transitions": (
                        group["paid_after_required_sweeps"]
                        - group["student_transition_end"]
                    ),
                    "sampled_groups": group["group"],
                    "optimizer_updates": group["optimizer_updates_after_group"],
                    "sweep_count": (
                        group["required_sweep_ordinals"][-1]
                        if group["required_sweep_ordinals"]
                        else (
                            group["selection_source_sweep_ordinal"]
                            if group["selection_source_sweep_ordinal"] is not None
                            else 0
                        )
                    ),
                    "crossed_regular_paid_thresholds": crossed,
                    "copied_from_evaluation_id": None,
                    "actor_parameter_fingerprint": group[
                        "actor_parameter_fingerprint_after_group"
                    ],
                }
            )
    expected_events.append(
        {
            "kind": "terminal",
            "paid_transitions": run["paid_transitions"],
            "student_transitions": run["student_transitions"],
            "probe_transitions": run["probe_transitions"],
            "sampled_groups": run["sampled_groups"],
            "optimizer_updates": run["optimizer_updates"],
            "sweep_count": run["probe_sweeps"],
            "crossed_regular_paid_thresholds": [],
            "copied_from_evaluation_id": None,
            "actor_parameter_fingerprint": run["final_parameter_fingerprint"],
        }
    )
    _require(
        len(records) == len(expected_events),
        "evaluation event count contains missing or extra records",
    )
    event_keys = tuple(expected_events[0])
    for index, (record, expected) in enumerate(zip(records, expected_events)):
        for key in event_keys:
            _require(
                record[key] == expected[key],
                f"evaluation event replay mismatch at {index} for {key}",
            )
    for sweep in sweeps:
        pre_id = sweep["pre_probe_evaluation_id"]
        post_id = sweep["post_probe_copy_evaluation_id"]
        _require(
            type(pre_id) is int
            and type(post_id) is int
            and post_id == pre_id + 1
            and records[pre_id]["paid_transitions"] == sweep["paid_before"]
            and records[post_id]["paid_transitions"] == sweep["paid_after"]
            and records[post_id]["copied_from_evaluation_id"] == pre_id,
            "sweep evaluation links mismatch",
        )
    return {
        "state_preserved": True,
        "plateaus_exact": True,
        "regular_schedule_exact": True,
        "event_sequence_exact": True,
        "same_actor_outputs_exact": True,
        "native_values": [record["native_success_rate"] for record in records],
    }


def _validate_groups(arm_name: str, run: dict) -> dict:
    groups = run.get("group_records")
    sweeps = run.get("probe_sweep_records")
    _require(
        isinstance(groups, list) and len(groups) == run["sampled_groups"],
        "group ledger length mismatch",
    )
    probed = arm_name in PROBED_ARMS
    if probed:
        current_paid = sweeps[0]["paid_after"]
        current_selection = sweeps[0]["selection_after_sweep"]
        current_sweep_ordinal = 1
        next_sweep_index = 1
    else:
        current_paid = 0
        current_selection = _recompute_selection(arm_name, None)
        current_sweep_ordinal = None
        next_sweep_index = 0
    current_student = 0
    current_updates = 0
    current_actor_update_calls = 0
    current_actor_applied_updates = 0
    previous_parameter_fingerprint = run["initial_parameter_fingerprint"]
    selection_rng = np.random.default_rng(run["rng_roots"]["selection"])
    student_reset_rng = np.random.default_rng(run["rng_roots"]["environment_reset_rng"])
    actor_action_rng = np.random.default_rng(run["rng_roots"]["actor_action"])
    task_groups = np.zeros(8, dtype=np.int64)
    task_rollouts = np.zeros(8, dtype=np.int64)
    task_successes = np.zeros(8, dtype=np.int64)
    task_transitions = np.zeros(8, dtype=np.int64)
    regimes = {"dead": 0, "mixed": 0, "all_pass": 0}
    zero_update_attempts = 0
    for group_index, group in enumerate(groups, start=1):
        _require_exact_keys(group, GROUP_KEYS, f"student group {group_index}")
        for key in (
            "actor_action_rng_fingerprint_before_student",
            "actor_action_rng_fingerprint_after_student",
            "environment_reset_rng_fingerprint_before_student",
            "environment_reset_rng_fingerprint_after_student",
        ):
            _require_sha256(group[key], f"student group {group_index} {key}")
        _require(
            group["actor_action_rng_fingerprint_before_student"]
            == _rng_state_fingerprint(actor_action_rng)
            and group["environment_reset_rng_fingerprint_before_student"]
            == _rng_state_fingerprint(student_reset_rng),
            "student action/reset RNG fingerprint before replay mismatch",
        )
        _require(group.get("group") == group_index, "group indices are not consecutive")
        _require(
            group.get("paid_before_group") == current_paid, "group paid start mismatch"
        )
        _require(
            group.get("student_transition_start") == current_student,
            "student start mismatch",
        )
        group_transitions = group.get("student_transitions")
        _require(
            type(group_transitions) is int
            and 1 <= group_transitions <= MAX_STUDENT_GROUP_TRANSITIONS,
            "student group transition bound failed",
        )
        student_after = current_student + group_transitions
        paid_after_student = current_paid + group_transitions
        _require(
            group.get("student_transition_end") == student_after
            and group.get("paid_after_student_group") == paid_after_student,
            "student group coordinate mismatch",
        )
        observed_probabilities = group.get("selection_probabilities_before_group")
        _require_vector_shape(
            observed_probabilities, 8, "group selection probabilities"
        )
        _require(
            _close(observed_probabilities, current_selection["probabilities"]),
            "group selection probabilities do not use latest sweep",
        )
        _require(
            group.get("selection_source_sweep_ordinal") == current_sweep_ordinal,
            "group selection source sweep mismatch",
        )
        for key, observed_key in (
            ("p_hat", "p_hat_used_before_group"),
            ("utility", "utility_used_before_group"),
            ("logits", "logits_used_before_group"),
        ):
            expected, observed = current_selection[key], group.get(observed_key)
            if expected is None:
                _require(observed is None, f"group unexpectedly has {observed_key}")
            else:
                _require(_close(observed, expected), f"group {observed_key} mismatch")
        task_id = group.get("task_id")
        _require(type(task_id) is int and 0 <= task_id < 8, "group task id invalid")
        _require(
            group.get("threshold") == THRESHOLDS[task_id], "group threshold mismatch"
        )
        _require(
            math.isclose(
                group.get("selected_task_probability"),
                current_selection["probabilities"][task_id],
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "selected task probability mismatch",
        )
        _require(
            group.get("selection_draw_index") == group_index - 1,
            "selection draw index mismatch",
        )
        expected_rng_before = _rng_state_fingerprint(selection_rng)
        expected_uniform = float(selection_rng.random())
        expected_rng_after = _rng_state_fingerprint(selection_rng)
        expected_task = int(
            min(
                np.searchsorted(
                    np.cumsum(np.asarray(current_selection["probabilities"])),
                    expected_uniform,
                    side="right",
                ),
                7,
            )
        )
        _require(
            group.get("selection_rng_fingerprint_before") == expected_rng_before
            and group.get("selection_rng_fingerprint_after") == expected_rng_after,
            "selection RNG fingerprint replay mismatch",
        )
        _require_sha256(expected_rng_before, "selection RNG before")
        _require_sha256(expected_rng_after, "selection RNG after")
        _require(
            math.isclose(
                group.get("selection_uniform"),
                expected_uniform,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and task_id == expected_task,
            "selection RNG uniform/task replay mismatch",
        )

        success_flags = group.get("student_success_flags")
        rollout_records = group.get("student_rollout_records")
        _require(
            isinstance(success_flags, list)
            and len(success_flags) == N_ROLLOUTS
            and all(type(value) is bool for value in success_flags),
            "student success-flag ledger invalid",
        )
        _require(
            isinstance(rollout_records, list) and len(rollout_records) == N_ROLLOUTS,
            "student rollout ledger length invalid",
        )
        rollout_transition_sum = 0
        for rollout_index, rollout in enumerate(rollout_records):
            _require_exact_keys(
                rollout,
                STUDENT_ROLLOUT_KEYS,
                f"student group {group_index} rollout {rollout_index}",
            )
            expected_draw_index = (group_index - 1) * N_ROLLOUTS + rollout_index
            expected_reset_seed = int(student_reset_rng.integers(0, 2**31 - 1))
            _require(
                rollout.get("rollout") == rollout_index
                and rollout.get("student_reset_draw_index") == expected_draw_index
                and rollout.get("reset_seed") == expected_reset_seed,
                "student reset-stream coordinate replay mismatch",
            )
            transitions = rollout.get("transitions")
            _require(
                type(transitions) is int
                and 1 <= transitions <= MAX_EPISODE_STEPS
                and rollout.get("action_count") == transitions,
                "student rollout transition/action count invalid",
            )
            _require_sha256(rollout.get("action_sha256"), "student action digest")
            # One scalar categorical draw consumes one uniform variate; its
            # state advance is independent of the probability values.
            actor_action_rng.random(transitions)
            max_height = rollout.get("max_height")
            _require(
                isinstance(max_height, (int, float))
                and not isinstance(max_height, bool)
                and math.isfinite(max_height)
                and -2.0 <= max_height <= 2.0,
                "student rollout max height invalid",
            )
            _require(
                rollout.get("success") is success_flags[rollout_index]
                and rollout["success"] is bool(max_height > THRESHOLDS[task_id]),
                "student strict threshold predicate mismatch",
            )
            rollout_transition_sum += transitions
        _require(
            group["actor_action_rng_fingerprint_after_student"]
            == _rng_state_fingerprint(actor_action_rng)
            and group["environment_reset_rng_fingerprint_after_student"]
            == _rng_state_fingerprint(student_reset_rng),
            "student action/reset RNG fingerprint after replay mismatch",
        )
        _require(
            rollout_transition_sum == group_transitions,
            "student per-rollout transition sum mismatch",
        )
        boundaries = (
            _crossed_boundaries(current_student, student_after) if probed else []
        )
        _require(
            group.get("required_crossed_boundaries") == boundaries,
            "group boundary list mismatch",
        )
        expected_ordinals = list(
            range(next_sweep_index + 1, next_sweep_index + len(boundaries) + 1)
        )
        _require(
            group.get("required_sweep_ordinals") == expected_ordinals,
            "required sweep ordinal list mismatch",
        )
        current_paid = paid_after_student
        for boundary in boundaries:
            _require(next_sweep_index < len(sweeps), "required sweep missing")
            sweep = sweeps[next_sweep_index]
            _require(
                sweep["sweep_ordinal"] == next_sweep_index + 1
                and sweep["crossed_boundary_student_transition"] == boundary
                and sweep["student_transitions"] == student_after
                and sweep["sampled_groups"] == group_index
                and sweep["optimizer_updates"] == current_updates
                and sweep["paid_before"] == current_paid,
                "refresh sweep/group mapping mismatch",
            )
            current_paid = sweep["paid_after"]
            current_selection = sweep["selection_after_sweep"]
            current_sweep_ordinal = sweep["sweep_ordinal"]
            next_sweep_index += 1
        _require(
            group.get("paid_after_required_sweeps") == current_paid,
            "group paid end mismatch",
        )

        success_count = group.get("success_count")
        _require(
            type(success_count) is int and 0 <= success_count <= N_ROLLOUTS,
            "success count invalid",
        )
        regime = (
            "dead"
            if success_count == 0
            else "all_pass"
            if success_count == N_ROLLOUTS
            else "mixed"
        )
        _require(group.get("regime") == regime, "group regime mismatch")
        _require(success_count == sum(success_flags), "student success sum mismatch")
        regimes[regime] += 1
        expected_mass = (
            0.0
            if regime != "mixed"
            else 2.0 * (N_ROLLOUTS - success_count) / N_ROLLOUTS
        )
        _require(
            math.isclose(
                group.get("realized_practical_maxrl_abs_coefficient_mass"),
                expected_mass,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "practical MaxRL mass mismatch",
        )
        update = _require_exact_keys(
            group.get("update"), UPDATE_KEYS, f"student group {group_index} update"
        )
        rewards = np.asarray(success_flags, dtype=np.float64)
        expected_weights = (
            np.zeros(N_ROLLOUTS)
            if success_count == 0
            else rewards / success_count - 1.0 / N_ROLLOUTS
        )
        _require(
            _close(update.get("weights"), expected_weights), "update weights mismatch"
        )
        _require(
            update.get("source") == "practical_dropped_group_maxrl"
            and update.get("eligible") is (regime == "mixed")
            and update.get("requested") is (regime == "mixed"),
            "update eligibility/request mismatch",
        )
        _require(
            math.isclose(
                update.get("weight_sum"), float(expected_weights.sum()), abs_tol=1e-12
            )
            and math.isclose(
                update.get("weight_l1"),
                float(np.abs(expected_weights).sum()),
                abs_tol=1e-12,
            ),
            "update weight aggregate mismatch",
        )
        for key in ("parameter_fingerprint_before", "parameter_fingerprint_after"):
            _require_sha256(update.get(key), f"update {key}")
        _require_sha256(
            group.get("actor_parameter_fingerprint_before_group"),
            "group actor fingerprint before",
        )
        _require_sha256(
            group.get("actor_parameter_fingerprint_after_group"),
            "group actor fingerprint after",
        )
        _require(
            group["actor_parameter_fingerprint_before_group"]
            == previous_parameter_fingerprint
            == update["parameter_fingerprint_before"],
            "actor fingerprint chain before update mismatch",
        )
        _require(
            group["actor_parameter_fingerprint_after_group"]
            == update["parameter_fingerprint_after"],
            "actor fingerprint chain after update mismatch",
        )
        _require(
            update.get("actor_update_calls_before") == current_actor_update_calls
            and update.get("actor_applied_updates_before")
            == current_actor_applied_updates,
            "actor update counters before mismatch",
        )
        if regime == "mixed":
            current_actor_update_calls += 1
            diagnostics = _require_exact_keys(
                update.get("diagnostics"),
                UPDATE_DIAGNOSTIC_KEYS,
                f"student group {group_index} update diagnostics",
            )
            _require(
                all(
                    isinstance(diagnostics[key], (int, float))
                    and math.isfinite(diagnostics[key])
                    for key in UPDATE_DIAGNOSTIC_KEYS
                ),
                "update diagnostics nonfinite",
            )
            _require(
                diagnostics["gradient_norm"] >= 0.0
                and diagnostics["update_norm"] >= 0.0
                and 0.0 <= diagnostics["mean_policy_entropy"] <= math.log(3.0),
                "update diagnostics outside registered bounds",
            )
            _require(
                math.isclose(
                    diagnostics["update_norm"],
                    3e-4 * diagnostics["gradient_norm"],
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ),
                "update norm does not match frozen learning rate",
            )
            expected_applied = diagnostics["update_norm"] != 0.0
            _require(
                update.get("applied") is expected_applied,
                "update applied flag mismatch",
            )
            if expected_applied:
                current_updates += 1
                current_actor_applied_updates += 1
                _require(
                    update["parameter_fingerprint_before"]
                    != update["parameter_fingerprint_after"],
                    "applied update left parameter fingerprint unchanged",
                )
            else:
                zero_update_attempts += 1
                _require(
                    update["parameter_fingerprint_before"]
                    == update["parameter_fingerprint_after"],
                    "zero update changed actor parameters",
                )
        else:
            _require(
                update.get("diagnostics") is None
                and update.get("applied") is False
                and update["parameter_fingerprint_before"]
                == update["parameter_fingerprint_after"],
                "ineligible group update record mismatch",
            )
        _require(
            update.get("actor_update_calls_after") == current_actor_update_calls
            and update.get("actor_applied_updates_after")
            == current_actor_applied_updates,
            "actor update counters after mismatch",
        )
        previous_parameter_fingerprint = update["parameter_fingerprint_after"]
        _require(
            group.get("optimizer_updates_after_group") == current_updates,
            "optimizer update ledger mismatch",
        )
        task_groups[task_id] += 1
        task_rollouts[task_id] += N_ROLLOUTS
        task_successes[task_id] += success_count
        task_transitions[task_id] += group_transitions
        current_student = student_after
    _require(next_sweep_index == len(sweeps), "unmapped probe sweep remains")
    _require(groups, "run must contain at least one student group")
    for index, group in enumerate(groups):
        _require(
            group["paid_before_group"] < run["paid_budget_nominal"],
            "extra student group started at/after paid budget",
        )
        if index < len(groups) - 1:
            _require(
                group["paid_after_required_sweeps"] < run["paid_budget_nominal"],
                "nonterminal group already exhausted paid budget",
            )
        else:
            _require(
                group["paid_after_required_sweeps"] >= run["paid_budget_nominal"],
                "terminal group did not reach paid budget",
            )
    _require(current_paid == run["paid_transitions"], "terminal paid ledger mismatch")
    _require(
        current_student == run["student_transitions"],
        "terminal student ledger mismatch",
    )
    _require(
        current_updates == run["optimizer_updates"],
        "terminal optimizer ledger mismatch",
    )
    _require(
        previous_parameter_fingerprint == run["final_parameter_fingerprint"],
        "terminal actor fingerprint mismatch",
    )
    _require(
        task_groups.tolist() == run["task_groups"], "task group aggregate mismatch"
    )
    _require(
        task_rollouts.tolist() == run["task_rollouts"],
        "task rollout aggregate mismatch",
    )
    _require(
        task_successes.tolist() == run["task_successes"],
        "task success aggregate mismatch",
    )
    _require(
        task_transitions.tolist() == run["task_student_transitions"],
        "task transition aggregate mismatch",
    )
    _require(regimes["mixed"] == run["live_groups"], "live group aggregate mismatch")
    _require(regimes["dead"] == run["dead_groups"], "dead group aggregate mismatch")
    _require(
        regimes["all_pass"] == run["all_pass_groups"],
        "all-pass group aggregate mismatch",
    )
    _require(
        zero_update_attempts == run["zero_gradient_update_attempts"],
        "zero update aggregate mismatch",
    )
    return {
        "ledger_valid": True,
        "regimes": regimes,
        "adaptive_nonuniform_in_group_selection": any(
            not np.array_equal(
                np.asarray(group["selection_probabilities_before_group"], dtype=float),
                np.full(8, 1.0 / 8.0),
            )
            for group in groups
        )
        if arm_name in ADAPTIVE_ARMS
        else False,
    }


def _normalized_truncated_auc(
    records: Sequence[dict], *, axis: str, cutoff: int
) -> float:
    x = np.asarray([record[axis] for record in records], dtype=np.float64)
    y = np.asarray(
        [record["target_uniform_mean_success"] for record in records],
        dtype=np.float64,
    )
    _require(
        len(x) >= 2 and x[0] == 0.0 and x[-1] >= cutoff,
        "AUC curve lacks cutoff coverage",
    )
    _require(np.all(np.diff(x) >= 0.0), "AUC axis is not monotone")
    _require(np.isfinite(x).all() and np.isfinite(y).all(), "AUC curve is nonfinite")
    area = 0.0
    for left in range(len(x) - 1):
        x0, x1 = float(x[left]), float(x[left + 1])
        y0, y1 = float(y[left]), float(y[left + 1])
        if x0 >= cutoff:
            break
        if x1 <= cutoff:
            area += 0.5 * (y0 + y1) * (x1 - x0)
            continue
        if x1 == x0:
            continue
        fraction = (cutoff - x0) / (x1 - x0)
        y_cutoff = y0 + fraction * (y1 - y0)
        area += 0.5 * (y0 + y_cutoff) * (cutoff - x0)
        break
    return float(area / cutoff)


def _normalized_full_auc(records: Sequence[dict], *, axis: str) -> float:
    x = np.asarray([record[axis] for record in records], dtype=np.float64)
    y = np.asarray(
        [record["target_uniform_mean_success"] for record in records],
        dtype=np.float64,
    )
    _require(len(x) >= 2 and np.all(np.diff(x) >= 0.0), "full AUC axis invalid")
    if x[-1] == x[0]:
        return float(y[-1])
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def _all_floats_finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_floats_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_floats_finite(item) for item in value)
    return True


def _validate_run(
    arm_name: str, run: dict, *, mode: str, budget: int, eval_n: int
) -> dict:
    _require_exact_keys(run, RUN_KEYS, f"{arm_name} run")
    seed = run.get("seed")
    _require(type(seed) is int, "run seed type invalid")
    _require(
        run.get("numeric_valid") is True and _all_floats_finite(run),
        "run numeric validity failed",
    )
    expected_domain = _rng_domain_record(seed)
    for key in (
        "logical_seed",
        "engine_master_seed",
        "environment_adapter_seed_argument",
        "rng_roots",
    ):
        _require(
            run.get(key) == expected_domain[key], f"run RNG domain mismatch for {key}"
        )
    _require(run.get("paid_budget_nominal") == budget, "run paid budget mismatch")
    _require(
        run.get("paid_transitions")
        == run.get("student_transitions") + run.get("probe_transitions"),
        "run paid identity mismatch",
    )
    _require(
        run.get("paid_budget_overshoot") == run.get("paid_transitions") - budget
        and run.get("paid_transitions") >= budget,
        "run paid overshoot mismatch",
    )
    _require(
        run.get("paid_transitions")
        <= budget + MAX_STUDENT_GROUP_TRANSITIONS + 2 * MAX_PROBE_SWEEP_TRANSITIONS,
        "run paid overshoot exceeds atomic bound",
    )
    _require(
        math.isclose(
            run.get("probe_fraction_of_paid"),
            run.get("probe_transitions") / run.get("paid_transitions"),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "probe paid fraction mismatch",
    )
    _require(
        run.get("student_rollouts") == run.get("sampled_groups") * N_ROLLOUTS,
        "student rollout count mismatch",
    )
    _require(
        run.get("probe_episodes")
        == run.get("probe_sweeps") * len(THRESHOLDS) * PROBES_PER_TASK,
        "probe episode count mismatch",
    )
    _require(
        run["probe_sweeps"] == len(run["probe_sweep_records"]),
        "probe_sweeps does not equal probe ledger length",
    )
    _require(
        run.get("total_parameters") == 640
        and run.get("active_parameters_per_task") == 640,
        "actor parameter contract mismatch",
    )
    _require_sha256(run["initial_parameter_fingerprint"], "initial actor fingerprint")
    _require_sha256(run["final_parameter_fingerprint"], "final actor fingerprint")
    for key in (
        "task_groups",
        "task_rollouts",
        "task_successes",
        "task_student_transitions",
    ):
        values = run[key]
        _require(
            isinstance(values, list)
            and len(values) == 8
            and all(type(value) is int and value >= 0 for value in values),
            f"run {key} vector invalid",
        )
    _require(
        isinstance(run["wall_seconds"], (int, float))
        and not isinstance(run["wall_seconds"], bool)
        and float(run["wall_seconds"]) > 0.0,
        "wall time invalid",
    )
    _require(
        isinstance(run["paid_transitions_per_wall_second"], (int, float))
        and not isinstance(run["paid_transitions_per_wall_second"], bool)
        and math.isclose(
            float(run["paid_transitions_per_wall_second"]),
            run["paid_transitions"] / float(run["wall_seconds"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "paid throughput mismatch",
    )
    _require(
        run.get("accounting_valid") is True
        and run.get("evaluation_rng_preserved") is True
        and run.get("probe_training_state_preserved") is True,
        "runner validity flags failed",
    )
    sweep_diagnostics = _validate_sweeps(arm_name, run)
    group_diagnostics = _validate_groups(arm_name, run)
    evaluation_diagnostics = _validate_evaluations(arm_name, run, eval_n)
    selection = _require_exact_keys(
        run["selection_diagnostics"],
        SELECTION_DIAGNOSTIC_KEYS,
        "selection diagnostics",
    )
    selection_matrix = np.asarray(
        [
            group["selection_probabilities_before_group"]
            for group in run["group_records"]
        ],
        dtype=np.float64,
    )
    _require(
        selection_matrix.shape == (run["sampled_groups"], 8)
        and np.isfinite(selection_matrix).all(),
        "selection diagnostic source matrix invalid",
    )
    realized = np.asarray(run["task_groups"], dtype=np.float64) / run["sampled_groups"]
    expected_selection = {
        "mean_selection_entropy": float(
            np.mean(
                -np.sum(
                    selection_matrix * np.log(np.maximum(selection_matrix, 1e-300)),
                    axis=1,
                )
            )
        ),
        "mean_selection_tv_from_uniform": float(
            np.mean(0.5 * np.abs(selection_matrix - 1.0 / 8.0).sum(axis=1))
        ),
        "mean_max_task_probability": float(np.mean(np.max(selection_matrix, axis=1))),
        "mean_assigned_probability_per_task": np.mean(
            selection_matrix, axis=0
        ).tolist(),
        "realized_task_fraction": realized.tolist(),
        "realized_task_tv_from_uniform": float(
            0.5 * np.abs(realized - 1.0 / 8.0).sum()
        ),
        "student_fraction_of_paid": (
            run["student_transitions"] / run["paid_transitions"]
        ),
        "probe_fraction_of_paid": (run["probe_transitions"] / run["paid_transitions"]),
        "paid_budget_overshoot": run["paid_transitions"] - budget,
    }
    for key, expected in expected_selection.items():
        observed = selection[key]
        if isinstance(expected, list):
            _require_vector_shape(observed, 8, f"selection diagnostic {key}")
            _require(_close(observed, expected), f"selection diagnostic {key} mismatch")
        elif key == "paid_budget_overshoot":
            _require(observed == expected, f"selection diagnostic {key} mismatch")
        else:
            _require(
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and math.isclose(
                    float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
                ),
                f"selection diagnostic {key} mismatch",
            )
    _require(
        math.isclose(
            selection["student_fraction_of_paid"] + selection["probe_fraction_of_paid"],
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "student/probe paid fractions do not sum to one",
    )
    records = run["evaluation_records"]
    fixed_paid_auc = _normalized_truncated_auc(
        records, axis="paid_transitions", cutoff=budget
    )
    full_paid_auc = _normalized_full_auc(records, axis="paid_transitions")
    student_auc = _normalized_full_auc(records, axis="student_transitions")
    for value in (fixed_paid_auc, full_paid_auc, student_auc):
        _require(0.0 <= value <= 1.0, "derived AUC outside [0,1]")
    return {
        "seed": seed,
        "raw": run,
        "derived": {
            "auc_target_uniform_mean_success_fixed_paid_budget": fixed_paid_auc,
            "auc_target_uniform_mean_success_full_atomic_paid": full_paid_auc,
            "auc_target_uniform_mean_success_by_student_transitions": student_auc,
            "fixed_minus_full_atomic_auc": fixed_paid_auc - full_paid_auc,
            "final_target_uniform_mean_success": records[-1][
                "target_uniform_mean_success"
            ],
            "final_native_success_rate": records[-1]["native_success_rate"],
        },
        "sweep_diagnostics": sweep_diagnostics,
        "group_diagnostics": group_diagnostics,
        "evaluation_diagnostics": evaluation_diagnostics,
        "strict_valid": True,
    }


def _recompute_case_summary(runs: list[dict]) -> dict:
    return {
        "n_attempted": len(runs),
        "n_valid": len(runs),
        "n_failed": 0,
        "ledger_means_descriptive_only": {
            key: float(np.mean([run[key] for run in runs]))
            for key in (
                "paid_transitions",
                "student_transitions",
                "probe_transitions",
                "probe_sweeps",
                "optimizer_updates",
            )
        },
    }


def _validate_cross_arm_crn(
    by_case: dict[str, list[dict]], seeds: Sequence[int]
) -> dict:
    by_arm_seed = {
        arm: {record["seed"]: record["raw"] for record in by_case[arm]}
        for arm in ARM_NAMES
    }
    actual_eval_by_actor: dict[tuple[int, str], dict] = {}
    for seed in seeds:
        runs = [by_arm_seed[arm][seed] for arm in ARM_NAMES]
        initial = runs[0]["evaluation_records"][0]
        for run in runs[1:]:
            observed = run["evaluation_records"][0]
            _require(
                observed["actor_parameter_fingerprint"]
                == initial["actor_parameter_fingerprint"]
                and observed["episode_records"] == initial["episode_records"]
                and all(observed[key] == initial[key] for key in SCORE_KEYS),
                "cross-arm initial evaluation CRN invariant failed",
            )

        max_groups = max(len(run["group_records"]) for run in runs)
        for draw_index in range(max_groups):
            present = [
                run["group_records"][draw_index]
                for run in runs
                if draw_index < len(run["group_records"])
            ]
            reference = present[0]
            _require(
                all(
                    group["selection_draw_index"] == draw_index
                    and group["selection_uniform"] == reference["selection_uniform"]
                    and group["selection_rng_fingerprint_before"]
                    == reference["selection_rng_fingerprint_before"]
                    and group["selection_rng_fingerprint_after"]
                    == reference["selection_rng_fingerprint_after"]
                    for group in present
                ),
                "cross-arm selection RNG pairing invariant failed",
            )
            for rollout_index in range(N_ROLLOUTS):
                rollout_reference = reference["student_rollout_records"][rollout_index]
                _require(
                    all(
                        group["student_rollout_records"][rollout_index][
                            "student_reset_draw_index"
                        ]
                        == rollout_reference["student_reset_draw_index"]
                        and group["student_rollout_records"][rollout_index][
                            "reset_seed"
                        ]
                        == rollout_reference["reset_seed"]
                        for group in present
                    ),
                    "cross-arm student reset-stream pairing invariant failed",
                )

        probed_runs = [by_arm_seed[arm][seed] for arm in PROBED_ARMS]
        max_sweeps = max(len(run["probe_sweep_records"]) for run in probed_runs)
        for sweep_index in range(max_sweeps):
            present = [
                run["probe_sweep_records"][sweep_index]
                for run in probed_runs
                if sweep_index < len(run["probe_sweep_records"])
            ]
            for task_id in range(8):
                for episode_index in range(PROBES_PER_TASK):
                    reference = present[0]["task_records"][task_id]["episodes"][
                        episode_index
                    ]
                    _require(
                        all(
                            sweep["task_records"][task_id]["episodes"][episode_index][
                                "reset_seed"
                            ]
                            == reference["reset_seed"]
                            and sweep["task_records"][task_id]["episodes"][
                                episode_index
                            ]["action_seed"]
                            == reference["action_seed"]
                            for sweep in present
                        ),
                        "cross-arm probe coordinate pairing invariant failed",
                    )

        sham = by_arm_seed["probe_sham_uniform_f5120"][seed]["group_records"]
        ordinary = by_arm_seed["ordinary_uniform"][seed]["group_records"]
        for sham_group, ordinary_group in zip(sham, ordinary):
            _require(
                sham_group["task_id"] == ordinary_group["task_id"]
                and sham_group["student_success_flags"]
                == ordinary_group["student_success_flags"]
                and sham_group["student_rollout_records"]
                == ordinary_group["student_rollout_records"]
                and sham_group["actor_parameter_fingerprint_before_group"]
                == ordinary_group["actor_parameter_fingerprint_before_group"]
                and sham_group["actor_parameter_fingerprint_after_group"]
                == ordinary_group["actor_parameter_fingerprint_after_group"]
                and sham_group["update"] == ordinary_group["update"],
                "sham/ordinary overlapping uniform mechanics diverged",
            )

        for run in runs:
            for evaluation in run["evaluation_records"]:
                if not evaluation["evaluation_was_executed"]:
                    continue
                key = (seed, evaluation["actor_parameter_fingerprint"])
                snapshot = {
                    "episode_records": evaluation["episode_records"],
                    **{score: evaluation[score] for score in SCORE_KEYS},
                }
                if key in actual_eval_by_actor:
                    _require(
                        snapshot == actual_eval_by_actor[key],
                        "cross-arm same-actor evaluation output invariant failed",
                    )
                else:
                    actual_eval_by_actor[key] = snapshot
    return {
        "passed": True,
        "paired_seeds_checked": list(seeds),
        "selection_rng_stream_replayed": True,
        "student_reset_stream_paired": True,
        "probe_coordinates_paired": True,
        "uniform_mechanics_paired_on_overlap": True,
        "same_actor_evaluations_paired": True,
    }


def validate_raw_artifact(raw: dict) -> dict:
    _require_exact_keys(raw, RAW_KEYS, "raw artifact")
    _require(_all_floats_finite(raw), "raw artifact contains nonfinite values")
    _require(raw["schema"] == RAW_SCHEMA, "raw artifact schema mismatch")
    _require(raw["artifact_state"] == "complete", "raw artifact is not complete")
    _require(raw["run_failures"] == [], "raw artifact contains failures")

    provenance = _require_exact_keys(raw["provenance"], PROVENANCE_KEYS, "provenance")
    _require_exact_keys(provenance["runtime"], RUNTIME_KEYS, "provenance runtime")
    _require(
        {key: provenance["runtime"][key] for key in PINNED_RUNTIME_VERSIONS}
        == PINNED_RUNTIME_VERSIONS,
        "raw provenance runtime is not the exact pinned runtime",
    )
    _require(_is_utc_iso8601(provenance["created_utc"]), "provenance timestamp invalid")
    _require(
        provenance["upstream_procurl_commit"] == UPSTREAM_PROCURL_COMMIT
        and provenance["upstream_code_copied"] is False
        and provenance["public_preexecution_registration"] is False,
        "provenance source/registration disclosure mismatch",
    )
    _require(
        provenance["seed_collision_audit"] == _independent_seed_collision_audit(),
        "provenance seed collision audit mismatch",
    )
    _require(
        isinstance(provenance["source_sha256"], dict)
        and set(provenance["source_sha256"]) == set(EXPECTED_SOURCE_RELATIVE_PATHS)
        and all(
            isinstance(relative, str)
            and relative
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for relative, digest in provenance["source_sha256"].items()
        ),
        "provenance source manifest invalid",
    )
    _require(
        provenance["source_lock_relative_path"]
        == "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_LOCK.json",
        "provenance source lock path is not canonical",
    )
    _require(
        provenance["git_commit"] is None
        or (
            isinstance(provenance["git_commit"], str)
            and len(provenance["git_commit"]) == 40
            and all(
                character in "0123456789abcdef"
                for character in provenance["git_commit"]
            )
        ),
        "provenance git commit invalid",
    )
    _require(
        isinstance(provenance["git_status_porcelain"], list)
        and all(isinstance(line, str) for line in provenance["git_status_porcelain"]),
        "provenance git status invalid",
    )

    protocol = raw["protocol"]
    _require(isinstance(protocol, dict), "protocol must be an object")
    mode = protocol.get("mode")
    seeds, budget, eval_n = _mode_schedule(mode)
    binding = protocol.get("development_gate")
    if mode == "confirmatory":
        _require_exact_keys(
            binding, BINDING_KEYS, "confirmation development gate binding"
        )
        _require(
            binding["all_gates_passed"] is True, "bound development gate did not pass"
        )
        for key in ("sha256", "raw_artifact_sha256"):
            _require_sha256(binding[key], f"development gate binding {key}")
        for key in ("relative_path", "raw_artifact_relative_path"):
            _require(
                isinstance(binding[key], str) and binding[key],
                f"development gate binding {key} invalid",
            )
    else:
        _require(binding is None, "development gate binding is confirmatory-only")
    _require(
        protocol == _independent_protocol(mode, binding),
        "protocol schema or frozen value mismatch",
    )

    if mode == "quick":
        _require(
            provenance["source_lock_enforced"] is False
            and provenance["source_lock_sha256"] is None,
            "quick provenance must not claim source-lock enforcement",
        )
    else:
        _require(
            provenance["source_lock_enforced"] is True,
            "nonquick raw did not enforce source lock",
        )
        _require_sha256(provenance["source_lock_sha256"], "raw source lock")

    cases = raw["cases"]
    _require(
        isinstance(cases, dict) and tuple(cases) == ARM_NAMES,
        "case key/order mismatch",
    )
    by_case: dict[str, list[dict]] = {}
    for arm_name in ARM_NAMES:
        case = _require_exact_keys(cases[arm_name], CASE_KEYS, f"case {arm_name}")
        config = _require_exact_keys(
            case["config"], CONFIG_KEYS, f"case {arm_name} config"
        )
        _require(config == ARM_CONFIGS[arm_name], "case config mismatch")
        runs = case["runs"]
        _require(
            isinstance(runs, list) and [run.get("seed") for run in runs] == list(seeds),
            "case run seed order mismatch",
        )
        summary = _require_exact_keys(
            case["summary"], SUMMARY_KEYS, f"case {arm_name} summary"
        )
        _require_exact_keys(
            summary["ledger_means_descriptive_only"],
            SUMMARY_LEDGER_KEYS,
            f"case {arm_name} ledger summary",
        )
        _require(
            summary == _recompute_case_summary(runs),
            "case summary does not exactly recompute",
        )
        by_case[arm_name] = [
            _validate_run(arm_name, run, mode=mode, budget=budget, eval_n=eval_n)
            for run in runs
        ]
    crn = _validate_cross_arm_crn(by_case, seeds)
    return {
        "mode": mode,
        "seeds": list(seeds),
        "paid_budget": budget,
        "eval_n": eval_n,
        "by_case": by_case,
        "cross_arm_crn_invariants": crn,
        "strict_valid": True,
    }


def development_gates(
    validated: dict,
    source_verification: dict,
    *,
    raw_artifact_relative_path: str,
    raw_artifact_sha256: str,
) -> dict:
    _require(
        validated.get("mode") == "development",
        "development gates require development raw",
    )
    _require_exact_keys(
        source_verification,
        SOURCE_VERIFICATION_KEYS,
        "source lock verification",
    )
    _require(source_verification["passed"] is True, "source lock verification failed")
    _require_exact_keys(
        source_verification["runtime"], RUNTIME_KEYS, "source verification runtime"
    )
    _require(
        isinstance(raw_artifact_relative_path, str) and raw_artifact_relative_path,
        "development raw relative path invalid",
    )
    _require_sha256(raw_artifact_sha256, "development raw artifact hash")
    all_runs = [record for arm in ARM_NAMES for record in validated["by_case"][arm]]
    all_sweeps = [
        sweep
        for arm in PROBED_ARMS
        for record in validated["by_case"][arm]
        for sweep in record["raw"]["probe_sweep_records"]
    ]
    pooled_regimes = {
        regime: sum(
            record["group_diagnostics"]["regimes"][regime] for record in all_runs
        )
        for regime in ("dead", "mixed", "all_pass")
    }
    native_values = [
        value
        for record in all_runs
        for value in record["evaluation_diagnostics"]["native_values"]
    ]
    adaptive_nonuniform = {
        arm: any(
            record["sweep_diagnostics"]["adaptive_nonuniform_seen"]
            or record["group_diagnostics"]["adaptive_nonuniform_in_group_selection"]
            for record in validated["by_case"][arm]
        )
        for arm in sorted(ADAPTIVE_ARMS)
    }
    gates = {
        "all_runs_source_numeric_parameter_rng_ledger_valid": bool(
            source_verification.get("passed") is True
            and all(record["strict_valid"] for record in all_runs)
        ),
        "all_sweeps_exact_probe_count_and_bounded_transitions": bool(
            all(
                record["sweep_diagnostics"]["all_exact_probe_count_and_bounded"]
                for record in all_runs
            )
        ),
        "all_p_hat_values_are_multiples_of_0p05": bool(
            all(
                record["sweep_diagnostics"]["all_p_hat_multiple"] for record in all_runs
            )
        ),
        "initial_and_crossed_boundary_sweep_schedule_exact": bool(
            all(record["sweep_diagnostics"]["schedule_exact"] for record in all_runs)
        ),
        "probes_preserve_actor_optimizer_and_training_rng": bool(
            all(record["sweep_diagnostics"]["state_preserved"] for record in all_runs)
        ),
        "paid_equals_student_plus_probe": bool(
            all(
                record["raw"]["paid_transitions"]
                == record["raw"]["student_transitions"]
                + record["raw"]["probe_transitions"]
                for record in all_runs
            )
        ),
        "uniform_arms_exact_and_ordinary_has_no_probes": bool(
            all(
                np.array_equal(
                    np.asarray(
                        group["selection_probabilities_before_group"], dtype=float
                    ),
                    np.full(8, 1.0 / 8.0),
                )
                for arm in ("probe_sham_uniform_f5120", "ordinary_uniform")
                for record in validated["by_case"][arm]
                for group in record["raw"]["group_records"]
            )
            and all(
                record["raw"]["probe_sweeps"] == 0
                and record["raw"]["probe_transitions"] == 0
                for record in validated["by_case"]["ordinary_uniform"]
            )
        ),
        "adaptive_probabilities_recompute_and_nonuniform_once": bool(
            all(adaptive_nonuniform.values())
        ),
        "each_probed_run_has_20k_student_transitions_and_update": bool(
            all(
                record["raw"]["student_transitions"] >= 20_000
                and record["raw"]["optimizer_updates"] >= 1
                for arm in PROBED_ARMS
                for record in validated["by_case"][arm]
            )
        ),
        "pooled_dead_mixed_all_pass_regimes_observed": bool(
            all(pooled_regimes[regime] > 0 for regime in pooled_regimes)
        ),
        "pooled_native_evaluation_values_vary": bool(len(set(native_values)) > 1),
    }
    _require(tuple(gates) == DEVELOPMENT_GATE_NAMES, "development gate order drift")
    diagnostics = {
        "n_runs": len(all_runs),
        "n_probe_sweeps": len(all_sweeps),
        "minimum_probed_student_transitions": min(
            record["raw"]["student_transitions"]
            for arm in PROBED_ARMS
            for record in validated["by_case"][arm]
        ),
        "minimum_probed_optimizer_updates": min(
            record["raw"]["optimizer_updates"]
            for arm in PROBED_ARMS
            for record in validated["by_case"][arm]
        ),
        "pooled_regime_counts": pooled_regimes,
        "adaptive_nonuniform_seen": adaptive_nonuniform,
        "n_distinct_native_evaluation_values": len(set(native_values)),
        "arm_contrasts_computed": False,
    }
    result = {
        "schema": GATE_SCHEMA,
        "mode": "development",
        "all_gates_passed": all(gates.values()),
        "source_lock_sha256": source_verification["source_lock_sha256"],
        "source_lock_verification": source_verification,
        "gates": gates,
        "diagnostics": diagnostics,
        "gate_policy": DEVELOPMENT_GATE_POLICY,
        "raw_artifact_relative_path": raw_artifact_relative_path,
        "raw_artifact_sha256": raw_artifact_sha256,
    }
    _require_exact_keys(result, GATE_KEYS, "development gate result")
    return result


def _bound_project_file(relative: object, label: str) -> tuple[Path, str]:
    _require(
        isinstance(relative, str) and relative and "\\" not in relative,
        f"{label} path must be canonical project-relative POSIX text",
    )
    pure = PurePosixPath(relative)
    _require(
        not pure.is_absolute()
        and pure.as_posix() == relative
        and all(part not in {"", ".", ".."} for part in pure.parts),
        f"{label} path must be canonical and cannot traverse",
    )
    root = PROJECT_ROOT.resolve()
    path = root.joinpath(*pure.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes project root") from error
    _require(path.is_file(), f"{label} file is missing")
    return path, relative


def verify_confirmation_development_gate(
    confirmatory_raw: dict, lock: dict, lock_path: Path
) -> dict:
    """Revalidate and exactly recompute the development gate bound by confirmation."""
    _require_exact_keys(confirmatory_raw, RAW_KEYS, "confirmatory raw artifact")
    protocol = confirmatory_raw.get("protocol")
    _require(isinstance(protocol, dict), "confirmatory protocol missing")
    _require(
        protocol.get("mode") == "confirmatory", "gate binding requires confirmation raw"
    )
    binding = _require_exact_keys(
        protocol.get("development_gate"),
        BINDING_KEYS,
        "confirmation development gate binding",
    )
    _require(binding["all_gates_passed"] is True, "bound development gate did not pass")
    gate_path, gate_relative = _bound_project_file(
        binding["relative_path"], "development gate"
    )
    raw_path, raw_relative = _bound_project_file(
        binding["raw_artifact_relative_path"], "development raw"
    )
    _require_sha256(binding["sha256"], "bound development gate hash")
    _require_sha256(binding["raw_artifact_sha256"], "bound development raw hash")
    _require(
        _sha256(gate_path) == binding["sha256"],
        "bound development gate hash mismatch",
    )
    _require(
        _sha256(raw_path) == binding["raw_artifact_sha256"],
        "bound development raw hash mismatch",
    )
    gate = load_strict_json(gate_path, "development gate")
    _require_exact_keys(gate, GATE_KEYS, "development gate")
    development_raw = load_strict_json(raw_path, "development raw artifact")
    source_verification = verify_source_lock(development_raw, lock, lock_path)
    validated = validate_raw_artifact(development_raw)
    _require(validated["mode"] == "development", "bound raw is not development mode")
    recomputed = development_gates(
        validated,
        source_verification,
        raw_artifact_relative_path=raw_relative,
        raw_artifact_sha256=_sha256(raw_path),
    )
    _require(gate == recomputed, "development gate does not exactly recompute")
    _require(
        gate["all_gates_passed"] is True
        and all(gate["gates"].values())
        and gate["source_lock_sha256"]
        == confirmatory_raw["provenance"]["source_lock_sha256"],
        "development gate is not positive or uses a different source lock",
    )
    return {
        "passed": True,
        "binding_exact": True,
        "development_gate_relative_path": gate_relative,
        "development_gate_sha256": binding["sha256"],
        "development_raw_relative_path": raw_relative,
        "development_raw_sha256": binding["raw_artifact_sha256"],
        "same_source_lock": True,
        "raw_revalidated": True,
        "gate_recomputed_exactly": True,
        "all_gates_passed": True,
    }


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 400
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        twice = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + twice) * (a + twice))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / ((a + twice) * (qap + twice))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise RuntimeError("incomplete-beta continued fraction did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def paired_t_test(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    _require(
        array.ndim == 1 and len(array) >= 2 and np.isfinite(array).all(),
        "paired t input invalid",
    )
    n = len(array)
    mean = float(np.mean(array))
    sample_std = float(np.std(array, ddof=1))
    if sample_std == 0.0:
        statistic = 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
        p_value = 1.0 if mean == 0.0 else 0.0
    else:
        statistic = mean / (sample_std / math.sqrt(n))
        degrees = n - 1
        beta_x = degrees / (degrees + statistic * statistic)
        p_value = _regularized_incomplete_beta(degrees / 2.0, 0.5, beta_x)
    return {
        "n_pairs": n,
        "mean_contrast": mean,
        "sample_std": sample_std,
        "paired_t_statistic": float(statistic),
        "degrees_of_freedom": n - 1,
        "paired_t_p_two_sided": float(p_value),
    }


def paired_bootstrap_ci(
    values: Sequence[float], *, seed: int, n_resamples: int = BOOTSTRAP_RESAMPLES
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    _require(
        array.ndim == 1 and len(array) >= 1 and np.isfinite(array).all(),
        "bootstrap input invalid",
    )
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    chunk = 2_000
    for start in range(0, n_resamples, chunk):
        size = min(chunk, n_resamples - start)
        indices = rng.integers(0, len(array), size=(size, len(array)))
        means[start : start + size] = array[indices].mean(axis=1)
    return [
        float(value) for value in np.quantile(means, (0.025, 0.975), method="linear")
    ]


def monte_carlo_sign_flip_p(
    values: Sequence[float],
    *,
    seed: int,
    n_draws: int = SIGN_FLIP_MONTE_CARLO_DRAWS,
) -> float:
    array = np.asarray(values, dtype=np.float64)
    _require(
        array.ndim == 1 and len(array) >= 1 and np.isfinite(array).all(),
        "sign-flip input invalid",
    )
    observed = abs(float(np.mean(array)))
    rng = np.random.default_rng(seed)
    extreme = 0
    chunk = 10_000
    for start in range(0, n_draws, chunk):
        size = min(chunk, n_draws - start)
        signs = rng.integers(0, 2, size=(size, len(array)), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        statistics = np.abs(signs @ array / len(array))
        extreme += int(np.count_nonzero(statistics >= observed - 1e-15))
    return float((extreme + 1) / (n_draws + 1))


def _holm(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    ordered = sorted((float(value), key) for key, value in p_values.items())
    total = len(ordered)
    running = 0.0
    still_rejecting = True
    result = {}
    for rank, (p_value, key) in enumerate(ordered, start=1):
        multiplier = total - rank + 1
        running = max(running, multiplier * p_value)
        reject = still_rejecting and p_value <= alpha / multiplier
        if not reject:
            still_rejecting = False
        result[key] = {
            "raw_p": p_value,
            "holm_adjusted_p": min(running, 1.0),
            "reject_familywise_0.05": bool(reject),
        }
    return result


def _metric_by_seed(validated: dict, arm: str, metric: str) -> dict[int, float]:
    return {
        record["seed"]: float(record["derived"][metric])
        for record in validated["by_case"][arm]
    }


def _paired_values(validated: dict, left: str, right: str, metric: str) -> list[float]:
    left_values = _metric_by_seed(validated, left, metric)
    right_values = _metric_by_seed(validated, right, metric)
    _require(left_values.keys() == right_values.keys(), "paired seed set mismatch")
    return [left_values[seed] - right_values[seed] for seed in sorted(left_values)]


def _contrast_analysis(
    validated: dict,
    *,
    left: str,
    right: str,
    metric: str,
    bootstrap_seed: int,
    sign_flip_seed: int | None = None,
) -> dict:
    values = _paired_values(validated, left, right, metric)
    analysis = {
        "left": left,
        "right": right,
        "metric": metric,
        **paired_t_test(values),
        "mean_ci95_paired_seed_bootstrap": paired_bootstrap_ci(
            values, seed=bootstrap_seed
        ),
        "paired_seed_bootstrap_seed": bootstrap_seed,
        "paired_seed_bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "paired_seed_bootstrap_quantiles": [0.025, 0.975],
        "paired_seed_bootstrap_quantile_method": "linear",
        "per_seed_contrast": values,
    }
    if sign_flip_seed is not None:
        analysis["monte_carlo_paired_sign_flip_p_two_sided"] = monte_carlo_sign_flip_p(
            values, seed=sign_flip_seed
        )
        analysis["monte_carlo_sign_flip_draws"] = SIGN_FLIP_MONTE_CARLO_DRAWS
        analysis["monte_carlo_sign_flip_seed"] = sign_flip_seed
        analysis["monte_carlo_sign_flip_plus_one_correction"] = True
    return analysis


def confirmatory_analysis(
    validated: dict,
    source_verification: dict,
    development_gate_verification: dict | None,
) -> dict:
    _require(
        validated.get("mode") == "confirmatory",
        "confirmatory analysis requires confirmation raw",
    )
    _require_exact_keys(
        source_verification,
        SOURCE_VERIFICATION_KEYS,
        "confirmation source lock verification",
    )
    _require(source_verification["passed"] is True, "confirmation source lock failed")
    _require(
        isinstance(development_gate_verification, dict)
        and development_gate_verification.get("passed") is True
        and development_gate_verification.get("binding_exact") is True
        and development_gate_verification.get("same_source_lock") is True
        and development_gate_verification.get("raw_revalidated") is True
        and development_gate_verification.get("gate_recomputed_exactly") is True
        and development_gate_verification.get("all_gates_passed") is True,
        "confirmation requires positive independently recomputed development gate",
    )
    metric = "auc_target_uniform_mean_success_fixed_paid_budget"
    primary = _contrast_analysis(
        validated,
        left="u16_probe_range_matched_f5120",
        right="procurl_env_b20_f5120",
        metric=metric,
        bootstrap_seed=31_000,
        sign_flip_seed=31_001,
    )
    primary["sesoi"] = PRIMARY_SESOI
    primary["supported"] = bool(
        primary["mean_contrast"] >= PRIMARY_SESOI
        and primary["paired_t_p_two_sided"] <= 0.05
    )
    specs = {
        "procurl_minus_sham": (
            "procurl_env_b20_f5120",
            "probe_sham_uniform_f5120",
        ),
        "u16_minus_sham": (
            "u16_probe_range_matched_f5120",
            "probe_sham_uniform_f5120",
        ),
        "procurl_minus_ordinary": (
            "procurl_env_b20_f5120",
            "ordinary_uniform",
        ),
        "u16_minus_ordinary": (
            "u16_probe_range_matched_f5120",
            "ordinary_uniform",
        ),
        "sham_minus_ordinary": (
            "probe_sham_uniform_f5120",
            "ordinary_uniform",
        ),
    }
    secondary = {
        name: _contrast_analysis(
            validated,
            left=left,
            right=right,
            metric=metric,
            bootstrap_seed=31_100 + index,
        )
        for index, (name, (left, right)) in enumerate(specs.items())
    }
    adjusted = _holm(
        {name: record["paired_t_p_two_sided"] for name, record in secondary.items()}
    )
    for name, record in secondary.items():
        record.update(adjusted[name])
    arm_descriptives = {}
    for arm in ARM_NAMES:
        records = validated["by_case"][arm]
        arm_descriptives[arm] = {}
        for derived_metric in (
            metric,
            "auc_target_uniform_mean_success_full_atomic_paid",
            "auc_target_uniform_mean_success_by_student_transitions",
            "final_target_uniform_mean_success",
            "final_native_success_rate",
        ):
            values = np.asarray(
                [record["derived"][derived_metric] for record in records], dtype=float
            )
            arm_descriptives[arm][derived_metric] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
                "per_seed": values.tolist(),
            }
        for raw_metric in (
            "paid_transitions",
            "student_transitions",
            "probe_transitions",
            "probe_fraction_of_paid",
            "probe_sweeps",
            "optimizer_updates",
        ):
            values = np.asarray(
                [record["raw"][raw_metric] for record in records], dtype=float
            )
            arm_descriptives[arm][raw_metric] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
                "per_seed": values.tolist(),
            }
    return {
        "schema": ANALYSIS_SCHEMA,
        "mode": "confirmatory",
        "strict_validation_passed": True,
        "source_lock_verification": source_verification,
        "development_gate_binding_verification": development_gate_verification,
        "primary": primary,
        "secondary_holm_family": secondary,
        "secondary_multiplicity": {
            "method": "Holm step-down",
            "familywise_alpha": 0.05,
            "test": "two-sided paired t-test",
            "family": list(specs),
        },
        "arm_descriptives": arm_descriptives,
        "statistical_conventions": {
            "primary_alpha": 0.05,
            "primary_bootstrap_seed": 31_000,
            "primary_sign_flip_seed": 31_001,
            "secondary_bootstrap_seeds_in_family_order": list(range(31_100, 31_105)),
            "bootstrap_quantiles": [0.025, 0.975],
            "bootstrap_quantile_method": "linear",
            "holm_familywise_alpha": 0.05,
            "sign_flip_draws": SIGN_FLIP_MONTE_CARLO_DRAWS,
            "sign_flip_plus_one_correction": True,
            "auc_integration": "trapezoid",
            "auc_duplicate_policy": (
                "record order retained; zero width adds zero; last duplicate "
                "starts the next positive-width segment"
            ),
            "auc_cutoff_interpolation": "linear",
        },
    }


def quick_engineering_report(validated: dict) -> dict:
    _require(validated.get("mode") == "quick", "quick report requires quick raw")
    return {
        "schema": ANALYSIS_SCHEMA,
        "mode": "quick",
        "status": "engineering_only_no_scientific_inference",
        "strict_validation_passed": True,
        "source_lock_verification": None,
        "per_arm": {
            arm: {
                "seeds": [record["seed"] for record in validated["by_case"][arm]],
                "derived_metrics": [
                    record["derived"] for record in validated["by_case"][arm]
                ],
                "ledger": [
                    {
                        "paid_transitions": record["raw"]["paid_transitions"],
                        "student_transitions": record["raw"]["student_transitions"],
                        "probe_transitions": record["raw"]["probe_transitions"],
                        "probe_sweeps": record["raw"]["probe_sweeps"],
                        "optimizer_updates": record["raw"]["optimizer_updates"],
                    }
                    for record in validated["by_case"][arm]
                ],
            }
            for arm in ARM_NAMES
        },
    }


def _write_json(path: Path, payload: dict, *, overwrite: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _project_relative(path: Path, label: str) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError as error:
        raise ValueError(f"{label} must be inside the project") from error


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        raw_path = args.raw.resolve()
        if not raw_path.is_file():
            raise ValueError(f"raw artifact is missing: {raw_path}")
        raw = load_strict_json(raw_path, "raw artifact")
        validated = validate_raw_artifact(raw)
        mode = validated["mode"]
        if mode == "quick":
            report = quick_engineering_report(validated)
        else:
            lock_path = args.lock.resolve()
            if not lock_path.is_file():
                raise ValueError(f"source lock is missing: {lock_path}")
            lock = load_strict_json(lock_path, "source lock")
            source_verification = verify_source_lock(raw, lock, lock_path)
            if mode == "development":
                report = development_gates(
                    validated,
                    source_verification,
                    raw_artifact_relative_path=_project_relative(
                        raw_path, "raw artifact"
                    ),
                    raw_artifact_sha256=_sha256(raw_path),
                )
            else:
                gate_verification = verify_confirmation_development_gate(
                    raw, lock, lock_path
                )
                report = confirmatory_analysis(
                    validated, source_verification, gate_verification
                )
        if mode != "development":
            report["raw_artifact_relative_path"] = _project_relative(
                raw_path, "raw artifact"
            )
            report["raw_artifact_sha256"] = _sha256(raw_path)
        if args.output is None:
            suffix = "development_gates" if mode == "development" else "analysis"
            args.output = HERE / f"acrobot_procurl_selection_{suffix}.json"
        _write_json(args.output, report, overwrite=args.overwrite)
    except (
        ValueError,
        TypeError,
        RuntimeError,
        FileExistsError,
        json.JSONDecodeError,
    ) as error:
        parser.error(str(error))
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
