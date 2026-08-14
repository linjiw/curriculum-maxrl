"""Evidence-grade four-arm runner for the frozen BARN/Gazebo campaign.

Unlike :mod:`icra2027.navigation_campaign`, this module consumes the frozen
BARN manifest and split and runs the real CPU Gazebo adapter.  The teacher's
task is a difficulty *stratum*: the adapter samples one training course from
the requested stratum for each rollout group.  Held-out courses are evaluated
through a second adapter instance which shares only the policy.  Consequently
evaluation cannot advance the training course RNG, launch counter, or budget
counters.

The JSON shape intentionally extends (rather than changes) the smoke-run
schema consumed by :mod:`icra2027.analyze_campaign`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from frontier_rl import (
    FrontierTeacher,
    FrontierTrainer,
    LearnabilityTeacher,
    StagedDifficultyTeacher,
    TrainerConfig,
    UniformTeacher,
)
from frontier_rl.adapters.barn_gazebo import (
    BarnCourse,
    BarnGazeboSpace,
    load_courses,
)
from frontier_rl.evaluation import TaskEval, summarize, teacher_calibration
from icra2027.freeze_pool_split import stratified_split


ARM_NAMES = ("ours_uN", "uniform", "learnability", "staged")
DEFAULT_MANIFEST = Path("icra2027/barn_manifest.jsonl")
DEFAULT_ROBOT_SDF = Path("icra2027/assets/barn_diff_drive.sdf")
DEFAULT_PREREG = Path("icra2027/prereg_icra.md")
DEFAULT_ANALYZER = Path("icra2027/analyze_campaign.py")
DEFAULT_PROTOCOL = Path("icra2027/barn_protocol.json")
SMOKE_EVIDENCE_STATUS = "engineering_smoke_not_paper_evidence"
FULL_EVIDENCE_STATUS = "full_barn_campaign"
ABLATION_EVIDENCE_STATUS = "full_barn_n_ablation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a 64-character SHA-256 digest")
    return text


def _verify_frozen_prereg(path: Path, expected_sha256: str) -> str:
    """Require a content-addressed preregistration with literal FROZEN status."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = _require_sha256(
        expected_sha256, field="expected_prereg_sha256")
    actual = _sha256(path)
    if not hmac.compare_digest(actual, expected):
        raise ValueError(
            f"preregistration hash mismatch: expected {expected}, got {actual}")
    if re.search(
        r"^\*\*Status:\*\* FROZEN[ \t]*$",
            path.read_text(), re.MULTILINE) is None:
        raise ValueError("evidence runs require a preregistration marked FROZEN")
    return actual


def _verify_frozen_protocol(
    path: Path, expected_sha256: str, campaign_cell: str
) -> tuple[str, dict, dict]:
    """Load the machine-readable protocol and select one exact evidence cell."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = _require_sha256(
        expected_sha256, field="expected_protocol_sha256")
    actual = _sha256(path)
    if not hmac.compare_digest(actual, expected):
        raise ValueError(
            f"protocol hash mismatch: expected {expected}, got {actual}")
    try:
        protocol = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid machine-readable protocol: {error}") from error
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("unsupported machine-readable protocol schema")
    if protocol.get("status") != "FROZEN":
        raise ValueError("evidence runs require a protocol marked FROZEN")
    if protocol.get("domain") != "barn_gazebo_cpu_navigation":
        raise ValueError("machine-readable protocol has the wrong domain")

    shared = protocol.get("shared_training")
    environment = protocol.get("environment")
    dataset = protocol.get("dataset")
    if not all(isinstance(value, dict)
               for value in (shared, environment, dataset)):
        raise ValueError("protocol is missing shared/environment/dataset settings")
    if campaign_cell == "primary":
        cell = protocol.get("primary")
        n_rollouts = cell.get("n_rollouts") if isinstance(cell, dict) else None
    else:
        match = re.fullmatch(r"ablation_n(2|4|8|16)", campaign_cell)
        cell = protocol.get("ablation")
        n_rollouts = int(match.group(1)) if match is not None else None
        if (match is None or not isinstance(cell, dict)
                or campaign_cell not in cell.get("fresh_cell_names", [])):
            raise ValueError(f"unknown protocol campaign cell: {campaign_cell!r}")
        if n_rollouts not in cell.get("n_values", []):
            raise ValueError(
                f"campaign cell {campaign_cell!r} is not declared in protocol")
    if not isinstance(cell, dict):
        raise ValueError(f"protocol is missing campaign cell {campaign_cell!r}")

    contract = {
        "campaign_cell": campaign_cell,
        "protocol_id": protocol.get("protocol_id"),
        "evidence_status": cell.get("evidence_status"),
        "arms": cell.get("arms"),
        "execution_order_by_seed": cell.get("execution_order_by_seed"),
        "n_rollouts": n_rollouts,
        "seeds": shared.get("seeds"),
        "tasks_per_step": shared.get("tasks_per_step"),
        "eval_episodes": shared.get("eval_episodes"),
        "training_sim_step_budget": shared.get("training_sim_step_budget"),
        "eval_sim_step_interval": shared.get("eval_sim_step_interval"),
        "steps": shared.get("max_training_updates"),
        "eval_every": shared.get("eval_every"),
        "teacher_floor": shared.get("teacher_floor"),
        "teacher_decay": shared.get("teacher_decay"),
        "teacher_gamma": shared.get("teacher_gamma"),
        "staged_initial_strata": shared.get("staged_initial_strata"),
        "staged_promotion_threshold": shared.get("staged_promotion_threshold"),
        "staged_min_frontier_groups": shared.get("staged_min_frontier_groups"),
        "episode_timeout": environment.get("episode_timeout"),
        "max_step_size": environment.get("max_step_size"),
        "real_time_update_rate": environment.get("real_time_update_rate"),
    }
    required = set(contract) - {"campaign_cell"}
    missing = sorted(key for key in required if contract.get(key) is None)
    if missing:
        raise ValueError(f"protocol campaign contract is missing {missing}")
    return actual, protocol, contract


def _enforce_protocol_contract(contract: dict, actual: dict) -> None:
    """Fail closed when a full-evidence CLI/config differs from its cell."""

    for field, expected in contract.items():
        observed = actual.get(field)
        if isinstance(expected, list):
            valid = list(observed) == expected if observed is not None else False
        else:
            valid = observed == expected
        if not valid:
            raise ValueError(
                f"full-evidence {field}={observed!r} differs from frozen "
                f"protocol value {expected!r}")


def _execution_identity(
    *,
    campaign_id: str | None,
    attempt_id: str | None,
    submitted_utc: str | None,
    slurm_job_id: str | None,
    slurm_array_job_id: str | None,
    slurm_array_task_id: int | None,
) -> dict:
    """Validate scheduler identity used for blind, immutable retry selection."""

    values = {
        "campaign_id": campaign_id,
        "attempt_id": attempt_id,
        "submitted_utc": submitted_utc,
        "slurm_job_id": slurm_job_id,
        "slurm_array_job_id": slurm_array_job_id,
        "slurm_array_task_id": slurm_array_task_id,
    }
    missing = sorted(key for key, value in values.items() if not value)
    if missing:
        raise ValueError(
            f"full evidence requires execution identity fields {missing}")
    for field in ("campaign_id", "attempt_id"):
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", values[field]) is None:
            raise ValueError(f"unsafe {field}: {values[field]!r}")
    for field in ("slurm_job_id", "slurm_array_job_id"):
        if re.fullmatch(r"[0-9]+(?:_[0-9]+)?", values[field]) is None:
            raise ValueError(f"invalid {field}: {values[field]!r}")
    if (not isinstance(values["slurm_array_task_id"], int)
            or isinstance(values["slurm_array_task_id"], bool)
            or values["slurm_array_task_id"] < 0):
        raise ValueError("slurm_array_task_id must be a non-negative integer")
    try:
        submitted = datetime.fromisoformat(
            values["submitted_utc"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("submitted_utc must be an ISO-8601 timestamp") from error
    if submitted.tzinfo is None or submitted.utcoffset() is None:
        raise ValueError("submitted_utc must include an explicit UTC offset")
    canonical_submitted = submitted.astimezone(timezone.utc).isoformat()
    return {
        "campaign_id": values["campaign_id"],
        "attempt_id": values["attempt_id"],
        "submitted_utc": canonical_submitted,
        "slurm_job_id": values["slurm_job_id"],
        "slurm_array_job_id": values["slurm_array_job_id"],
        "slurm_array_task_id": values["slurm_array_task_id"],
    }


def _read_manifest(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid manifest JSON on line {line_number}: {error}") from error
        required = (
            "env_id", "barn_index", "difficulty", "asset", "asset_sha256",
            "path_asset", "path_sha256",
        )
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(
                f"manifest line {line_number} missing required fields {missing}")
        normalized = dict(row)
        normalized["env_id"] = str(normalized["env_id"])
        normalized["barn_index"] = int(normalized["barn_index"])
        normalized["difficulty"] = float(normalized["difficulty"])
        if not np.isfinite(normalized["difficulty"]):
            raise ValueError(
                f"manifest line {line_number} has non-finite difficulty")
        for field in ("asset", "path_asset"):
            normalized[field] = str(normalized[field])
        for field in ("asset_sha256", "path_sha256"):
            normalized[field] = _require_sha256(
                normalized[field], field=f"manifest line {line_number} {field}")
        rows.append(normalized)
    if not rows:
        raise ValueError("BARN manifest is empty")
    ids = [row["env_id"] for row in rows]
    indices = [row["barn_index"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest env_id values must be unique")
    if len(indices) != len(set(indices)):
        raise ValueError("manifest barn_index values must be unique")
    return rows


def _string_id_list(split: dict, field: str) -> list[str]:
    value = split.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str)
                                               for item in value):
        raise ValueError(f"split {field} must be a list of strings")
    if len(value) != len(set(value)):
        raise ValueError(f"split {field} contains duplicate ids")
    return list(value)


def load_frozen_inputs(manifest_path: Path, split_path: Path) -> tuple[dict, list[dict]]:
    """Load and cross-check a split cryptographically bound to its manifest.

    The split freezer stores the byte-level manifest digest in
    ``source_sha256``.  This function fails closed if that binding is absent,
    if either partition does not cover the manifest exactly, if embedded
    records differ, or if the recorded split settings no longer reproduce the
    frozen ID lists.
    """

    manifest_path = Path(manifest_path)
    split_path = Path(split_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    rows = _read_manifest(manifest_path)
    try:
        split = json.loads(split_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid frozen split JSON: {error}") from error
    if not isinstance(split, dict):
        raise ValueError("frozen split must be a JSON object")
    if split.get("schema_version") != 1:
        raise ValueError("unsupported or missing frozen split schema_version")

    actual_manifest_sha = _sha256(manifest_path)
    recorded_manifest_sha = _require_sha256(
        split.get("source_sha256"), field="split source_sha256")
    if not hmac.compare_digest(actual_manifest_sha, recorded_manifest_sha):
        raise ValueError(
            "frozen split/manifest hash mismatch: "
            f"split binds {recorded_manifest_sha}, got {actual_manifest_sha}")

    train_ids = _string_id_list(split, "train_ids")
    heldout_ids = _string_id_list(split, "heldout_ids")
    train_set, heldout_set = set(train_ids), set(heldout_ids)
    if train_set & heldout_set:
        raise ValueError("frozen train and held-out partitions overlap")
    manifest_ids = {row["env_id"] for row in rows}
    if train_set | heldout_set != manifest_ids:
        missing = sorted(manifest_ids - (train_set | heldout_set))
        unknown = sorted((train_set | heldout_set) - manifest_ids)
        raise ValueError(
            "frozen split does not partition the manifest exactly: "
            f"missing={missing}, unknown={unknown}")

    expected_counts = {
        "n_total": len(rows),
        "n_train": len(train_ids),
        "n_heldout": len(heldout_ids),
    }
    for field, expected in expected_counts.items():
        if split.get(field) != expected:
            raise ValueError(
                f"split {field}={split.get(field)!r}, expected {expected}")
    n_strata = split.get("n_strata")
    if not isinstance(n_strata, int) or n_strata < 1:
        raise ValueError("split n_strata must be a positive integer")
    if min(len(train_ids), len(heldout_ids)) < n_strata:
        raise ValueError("need at least one train and held-out course per stratum")

    by_id = {row["env_id"]: row for row in rows}
    embedded = split.get("records")
    if not isinstance(embedded, dict) or set(embedded) != manifest_ids:
        raise ValueError("split records must contain every manifest id exactly")
    for env_id, row in by_id.items():
        # JSON-decoded manifest rows and freezer records are expected to be
        # identical.  This also rejects difficulty or asset substitutions.
        if embedded[env_id] != row:
            raise ValueError(
                f"split record for {env_id!r} differs from bound manifest")

    try:
        holdout_fraction = float(split["holdout_fraction_requested"])
        split_seed = int(split["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("split is missing valid seed/holdout settings") from error
    expected_train, expected_heldout = stratified_split(
        rows, holdout_fraction=holdout_fraction,
        n_strata=n_strata, seed=split_seed)
    if train_ids != expected_train or heldout_ids != expected_heldout:
        raise ValueError(
            "frozen IDs do not reproduce from the recorded split settings")
    return split, rows


def _safe_dataset_path(dataset_root: Path, relative: str, *, field: str) -> Path:
    root = dataset_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"manifest {field} escapes dataset root: {relative!r}") from error
    return path


def verify_dataset_assets(dataset_root: Path, rows: Iterable[dict]) -> None:
    """Verify every adapter-consumed BARN asset against the bound manifest."""

    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    for row in rows:
        checks = [("asset", "asset_sha256"),
                  ("path_asset", "path_sha256")]
        if "grid_asset" in row or "grid_sha256" in row:
            if "grid_asset" not in row or "grid_sha256" not in row:
                raise ValueError(
                    f"manifest row {row['env_id']} has incomplete grid checksum fields")
            checks.append(("grid_asset", "grid_sha256"))
        for path_field, hash_field in checks:
            path = _safe_dataset_path(
                dataset_root, str(row[path_field]), field=path_field)
            if not path.is_file():
                raise FileNotFoundError(path)
            expected = _require_sha256(
                row[hash_field], field=f"{row['env_id']} {hash_field}")
            actual = _sha256(path)
            if not hmac.compare_digest(actual, expected):
                raise ValueError(
                    f"asset hash mismatch for {row['env_id']} {path_field}: "
                    f"expected {expected}, got {actual}")


def _derived_seed(namespace: int, seed: int, component: int = 0) -> int:
    if seed < 0:
        raise ValueError("campaign seed must be non-negative")
    state = np.random.SeedSequence([
        int(namespace) & 0xFFFFFFFF,
        int(seed) & 0xFFFFFFFF,
        (int(seed) >> 32) & 0xFFFFFFFF,
        int(component) & 0xFFFFFFFF,
    ]).generate_state(1, dtype=np.uint32)[0]
    # Gazebo Classic accepts a signed positive seed most portably.
    return int(state % np.uint32(2**31 - 1))


def make_teacher(
    arm: str,
    n_tasks: int,
    n_rollouts: int,
    seed: int,
    *,
    floor: float = 0.1,
    decay: float = 0.7,
    gamma: float = 1.0,
    staged_initial_strata: int = 1,
    staged_promotion_threshold: float = 0.7,
    staged_min_frontier_groups: int = 5,
):
    common = dict(
        n_tasks=n_tasks, n_rollouts=n_rollouts, floor=floor,
        decay=decay, seed=seed,
    )
    if arm == "ours_uN":
        return FrontierTeacher(**common, gamma=gamma)
    if arm == "uniform":
        return UniformTeacher(**common, gamma=gamma)
    if arm == "learnability":
        return LearnabilityTeacher(**common, gamma=gamma)
    if arm == "staged":
        return StagedDifficultyTeacher(
            **common,
            difficulty_order=np.arange(n_tasks),
            initial_tasks=staged_initial_strata,
            promotion_threshold=staged_promotion_threshold,
            min_frontier_groups=staged_min_frontier_groups,
        )
    raise ValueError(f"unknown arm {arm!r}; choose from {ARM_NAMES}")


def _teacher_diagnostics(teacher) -> dict:
    """Return diagnostics without drawing from the teacher's sampling RNG."""

    p_hat = teacher.pass_rate_estimates()
    if isinstance(teacher, UniformTeacher):
        weights = np.full(teacher.n_tasks, 1.0 / teacher.n_tasks)
    elif isinstance(teacher, StagedDifficultyTeacher):
        active = teacher.difficulty_order[:teacher.active_count]
        staged = np.zeros(teacher.n_tasks, dtype=float)
        staged[active] = 1.0 / len(active)
        weights = ((1.0 - teacher.floor) * staged
                   + teacher.floor / teacher.n_tasks)
    else:
        utility = teacher.utility(p_hat) ** teacher.gamma
        weights = (utility / utility.sum() if utility.sum() > 1e-12
                   else np.full(teacher.n_tasks, 1.0 / teacher.n_tasks))
        weights = ((1.0 - teacher.floor) * weights
                   + teacher.floor / teacher.n_tasks)
    return {
        "posterior_mean": p_hat.tolist(),
        "sampling_weights_at_posterior_mean": weights.tolist(),
        "visits": teacher.visits.tolist(),
        **teacher.metrics(),
    }


def _freeze_state(value):
    """Convert nested NumPy state into an equality-safe immutable value."""

    if isinstance(value, np.ndarray):
        return ("ndarray", str(value.dtype), tuple(value.shape), value.tobytes())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_state(item))
                            for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_state(item) for item in value)
    return copy.deepcopy(value)


def _training_state(env, teacher) -> dict:
    policy = env.policy
    policy_state = (policy.state_dict() if hasattr(policy, "state_dict")
                    else vars(policy))
    env_rng = (env.rng.bit_generator.state if hasattr(env, "rng") else None)
    teacher_rng = (teacher.rng.bit_generator.state
                   if hasattr(teacher, "rng") else None)
    return {
        "training_episodes": int(env.training_episodes),
        "training_sim_steps": int(env.training_sim_steps),
        "course_launches": getattr(env, "course_launches", None),
        "episode_counter": getattr(env, "_episode_counter", None),
        "training_rng": _freeze_state(env_rng),
        "teacher_state": _freeze_state(teacher.state_dict()),
        "teacher_rng": _freeze_state(teacher_rng),
        "policy_state": _freeze_state(policy_state),
        "policy_updates": getattr(policy, "updates", None),
    }


class _InstrumentedTrainingSpace:
    """Transparent trainer view which distinguishes all-fail/all-pass groups."""

    def __init__(self, env):
        self.env = env
        self.all_fail_groups = 0
        self.all_pass_groups = 0
        self.live_groups = 0
        self.policy_updates = 0
        self.sampled_stratum_counts: Counter[int] = Counter()
        self.training_status_counts: Counter[str] = Counter()
        self.training_course_counts: Counter[str] = Counter()
        self.training_episode_records: list[dict] = []

    @property
    def n_tasks(self) -> int:
        return self.env.n_tasks

    def rollout_group(self, task_id: int, n_rollouts: int):
        group = self.env.rollout_group(task_id, n_rollouts)
        rewards = np.asarray(group.rewards, dtype=float)
        if len(group.infos) != len(rewards):
            raise ValueError("rollout group infos/rewards length mismatch")
        self.sampled_stratum_counts[int(task_id)] += 1
        for episode_index, (reward, info) in enumerate(
                zip(rewards, group.infos)):
            status = str(info.get("status", "unknown"))
            course_id = str(info.get("course_id", "unknown"))
            self.training_status_counts[status] += 1
            self.training_course_counts[course_id] += 1
            self.training_episode_records.append({
                "episode_index": len(self.training_episode_records),
                "group_episode_index": episode_index,
                "stratum": int(task_id),
                "course_id": course_id,
                "difficulty": (float(info["difficulty"])
                               if info.get("difficulty") is not None else None),
                "success": int(bool(reward)),
                "status": status,
                "sim_steps": int(info.get("sim_steps", 0)),
                "sim_seconds": float(info.get("sim_seconds", 0.0)),
                "planned_clearance_m": (
                    float(info["planned_clearance_m"])
                    if info.get("planned_clearance_m") is not None else None),
            })
        successes = float(rewards.sum())
        if successes == 0.0:
            self.all_fail_groups += 1
        elif successes == len(rewards):
            self.all_pass_groups += 1
        else:
            self.live_groups += 1
        return group

    def relabel(self, group):
        return self.env.relabel(group)

    def update(self, task_id: int, trajectories, weights) -> None:
        self.env.update(task_id, trajectories, weights)
        self.policy_updates += 1


def _strata(courses: Sequence[BarnCourse], n_strata: int) -> list[list[BarnCourse]]:
    ordered = sorted(courses, key=lambda course: (course.difficulty, course.env_id))
    return [list(part) for part in np.array_split(
        np.asarray(ordered, dtype=object), n_strata)]


def _evaluate_heldout(
    eval_env,
    heldout_courses: Sequence[BarnCourse],
    teacher,
    *,
    n_strata: int,
    n_episodes: int,
    eval_seed: int,
) -> dict:
    if n_episodes < 1:
        raise ValueError("eval_episodes must be positive")
    ordered = sorted(
        heldout_courses, key=lambda course: (course.difficulty, course.env_id))
    stratum_for = {
        course.env_id: stratum_id
        for stratum_id, courses in enumerate(_strata(ordered, n_strata))
        for course in courses
    }
    evals = []
    rows = []
    episode_rows = []
    statuses: Counter[str] = Counter()
    for task_id, course in enumerate(ordered):
        course_seed = _derived_seed(0xE7A1C0DE, eval_seed, course.barn_index)
        result = eval_env.evaluate_course(
            course, n_episodes, seed=course_seed)
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError(
                "BarnGazeboSpace.evaluate_course must return "
                "(successes, sim_steps, episodes)")
        successes, sim_steps, episodes = result
        successes, sim_steps = int(successes), int(sim_steps)
        if not 0 <= successes <= n_episodes:
            raise ValueError(
                f"invalid success count for {course.env_id}: {successes}")
        if sim_steps < 0:
            raise ValueError(f"negative simulator steps for {course.env_id}")
        if len(episodes) != n_episodes:
            raise ValueError(
                f"{course.env_id} returned {len(episodes)} episode rows, "
                f"expected {n_episodes}")
        statuses.update(str(row.get("status", "unknown")) for row in episodes)
        for episode_index, episode in enumerate(episodes):
            episode_rows.append({
                "env_id": course.env_id,
                "barn_index": course.barn_index,
                "seed": course_seed,
                "episode_index": episode_index,
                "success": int(bool(episode.get(
                    "success", episode_index < successes))),
                "status": str(episode.get("status", "unknown")),
                "sim_steps": int(episode.get("sim_steps", 0)),
                "sim_seconds": float(episode.get("sim_seconds", 0.0)),
                "planned_clearance_m": (
                    float(episode["planned_clearance_m"])
                    if episode.get("planned_clearance_m") is not None else None),
            })
        evaluation = TaskEval(task_id, n_episodes, successes, sim_steps)
        evals.append(evaluation)
        rows.append({
            "env_id": course.env_id,
            "barn_index": course.barn_index,
            "difficulty": course.difficulty,
            "stratum": stratum_for[course.env_id],
            "seed": course_seed,
            "successes": successes,
            "episodes": n_episodes,
            "success_rate": evaluation.rate(),
            "sim_steps": sim_steps,
        })

    easy_count = max(1, int(np.ceil(len(ordered) / 10)))
    ks = tuple(k for k in (1, 4, 8) if k <= n_episodes)
    summary = summarize(
        evals, ks=ks, easy_set=list(range(easy_count)))
    rates = np.asarray([row["success_rate"] for row in rows], dtype=float)
    difficulty_bins = []
    for bin_id, indices in enumerate(np.array_split(
            np.arange(len(rows)), min(10, len(rows)))):
        selected = [rows[int(index)] for index in indices]
        difficulty_bins.append({
            "bin": bin_id,
            "n_courses": len(selected),
            "difficulty_min": float(selected[0]["difficulty"]),
            "difficulty_max": float(selected[-1]["difficulty"]),
            "mean_success": float(rates[indices].mean()),
        })

    stratum_evals = []
    for stratum_id in range(n_strata):
        selected = [row for row in rows if row["stratum"] == stratum_id]
        stratum_evals.append(TaskEval(
            stratum_id,
            sum(row["episodes"] for row in selected),
            sum(row["successes"] for row in selected),
            sum(row["sim_steps"] for row in selected),
        ))
    summary.update(teacher_calibration(teacher, stratum_evals, min_visits=1))
    summary.update({
        "per_task_success": rates.tolist(),
        "heldout_course_ids": [row["env_id"] for row in rows],
        "per_course": rows,
        "episode_records": episode_rows,
        "success_by_difficulty_bin": [
            row["mean_success"] for row in difficulty_bins],
        "difficulty_bins": difficulty_bins,
        "status_counts": dict(sorted(statuses.items())),
    })
    return summary


def _normalized_auc(
    history: Sequence[dict], currency: str, *, budget: float | None = None
) -> float:
    x = np.asarray([row[currency] for row in history], dtype=float)
    y = np.asarray([row["eval"]["mean_success"] for row in history], dtype=float)
    if len(x) == 0:
        raise ValueError("cannot compute AUC from empty history")
    if np.any(np.diff(x) < 0.0):
        raise ValueError(f"non-monotone {currency} history")
    if budget is not None:
        budget = float(budget)
        if budget <= 0.0 or x[-1] < budget:
            raise ValueError(
                f"{currency} history does not reach requested AUC budget")
        keep = x < budget
        clipped_x = x[keep]
        clipped_y = y[keep]
        boundary_y = float(np.interp(budget, x, y))
        x = np.concatenate((clipped_x, [budget]))
        y = np.concatenate((clipped_y, [boundary_y]))
    if x[-1] <= 0.0:
        return float(y[-1])
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y, x) / x[-1])


def _close_if_supported(env) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def run_one(
    arm: str,
    seed: int,
    *,
    train_courses: Sequence[BarnCourse],
    heldout_courses: Sequence[BarnCourse],
    n_strata: int,
    robot_sdf: Path,
    stepper_path: Path | None,
    runtime_root: Path,
    steps: int = 60,
    n_rollouts: int = 16,
    tasks_per_step: int = 4,
    eval_every: int = 10,
    eval_episodes: int = 8,
    episode_timeout: float = 25.0,
    max_step_size: float = 0.005,
    real_time_update_rate: int = 2000,
    domain_id: int = 80,
    master_port: int = 11500,
    teacher_floor: float = 0.1,
    teacher_decay: float = 0.7,
    teacher_gamma: float = 1.0,
    staged_initial_strata: int = 1,
    staged_promotion_threshold: float = 0.7,
    staged_min_frontier_groups: int = 5,
    training_sim_step_budget: int | None = None,
    eval_sim_step_interval: int | None = None,
) -> dict:
    """Run one paired training seed/arm with an isolated held-out adapter."""

    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm {arm!r}")
    if steps < 1 or eval_every < 1 or n_rollouts < 2 or tasks_per_step < 1:
        raise ValueError(
            "require steps/eval_every/tasks_per_step >= 1 and n_rollouts >= 2")
    if training_sim_step_budget is not None and training_sim_step_budget < 1:
        raise ValueError("training_sim_step_budget must be positive")
    if eval_sim_step_interval is not None and eval_sim_step_interval < 1:
        raise ValueError("eval_sim_step_interval must be positive")
    if ((training_sim_step_budget is None)
            != (eval_sim_step_interval is None)):
        raise ValueError(
            "transition-budget runs require both training_sim_step_budget "
            "and eval_sim_step_interval")
    if len(train_courses) < n_strata or len(heldout_courses) < n_strata:
        raise ValueError("need at least one train and held-out course per stratum")

    teacher_seed = _derived_seed(0x7EAC4E12, seed)
    eval_seed = _derived_seed(0xE7A15EED, seed)
    train_env = BarnGazeboSpace(
        list(train_courses), robot_sdf, Path(runtime_root) / arm / "train",
        seed=seed, n_strata=n_strata, domain_id=domain_id,
        master_port=master_port, episode_timeout=episode_timeout,
        max_step_size=max_step_size,
        real_time_update_rate=real_time_update_rate,
        stepper_path=stepper_path,
    )
    eval_env = BarnGazeboSpace(
        list(heldout_courses), robot_sdf, Path(runtime_root) / arm / "eval",
        seed=eval_seed, n_strata=n_strata, domain_id=domain_id + 1,
        master_port=master_port + 1, episode_timeout=episode_timeout,
        max_step_size=max_step_size,
        real_time_update_rate=real_time_update_rate,
        policy=train_env.policy,
        stepper_path=stepper_path,
    )
    teacher = make_teacher(
        arm, train_env.n_tasks, n_rollouts, teacher_seed,
        floor=teacher_floor, decay=teacher_decay, gamma=teacher_gamma,
        staged_initial_strata=staged_initial_strata,
        staged_promotion_threshold=staged_promotion_threshold,
        staged_min_frontier_groups=staged_min_frontier_groups,
    )
    instrumented = _InstrumentedTrainingSpace(train_env)
    trainer = FrontierTrainer(
        instrumented,
        instrumented,
        TrainerConfig(
            n_rollouts=n_rollouts,
            tasks_per_step=tasks_per_step,
            hindsight=False,
            estimator="maxrl",
            teacher_gamma=teacher_gamma,
            teacher_decay=teacher_decay,
            teacher_floor=teacher_floor,
            seed=seed,
        ),
        teacher=teacher,
    )

    history = []
    totals = {
        "live_groups": 0,
        "dead_groups": 0,
        "relabeled_groups": 0,
        "training_wall_seconds": 0.0,
        "evaluation_wall_seconds": 0.0,
    }

    def checkpoint(step: int) -> None:
        before = _training_state(train_env, teacher)
        started = time.perf_counter()
        result = _evaluate_heldout(
            eval_env, heldout_courses, teacher,
            n_strata=n_strata, n_episodes=eval_episodes,
            eval_seed=eval_seed)
        eval_wall = time.perf_counter() - started
        after = _training_state(train_env, teacher)
        if after != before:
            changed = sorted(key for key in before if before[key] != after[key])
            raise RuntimeError(
                "held-out evaluation mutated training state: "
                + ", ".join(changed))
        totals["evaluation_wall_seconds"] += eval_wall
        groups = totals["live_groups"] + totals["dead_groups"]
        history.append({
            "step": step,
            "episodes": int(train_env.training_episodes),
            "sim_steps": int(train_env.training_sim_steps),
            "training_wall_seconds": totals["training_wall_seconds"],
            "evaluation_wall_seconds": totals["evaluation_wall_seconds"],
            "dead_group_rate": totals["dead_groups"] / max(groups, 1),
            "all_fail_group_rate": instrumented.all_fail_groups / max(groups, 1),
            "all_pass_group_rate": instrumented.all_pass_groups / max(groups, 1),
            "live_groups": totals["live_groups"],
            "dead_groups": totals["dead_groups"],
            "all_fail_groups": instrumented.all_fail_groups,
            "all_pass_groups": instrumented.all_pass_groups,
            "relabeled_groups": totals["relabeled_groups"],
            "updates_live": instrumented.policy_updates,
            "updates_relabel": 0,
            "sampled_stratum_counts": {
                str(key): value for key, value in
                sorted(instrumented.sampled_stratum_counts.items())},
            "training_status_counts": dict(sorted(
                instrumented.training_status_counts.items())),
            "training_course_counts": dict(sorted(
                instrumented.training_course_counts.items())),
            "eval": result,
            "teacher": _teacher_diagnostics(teacher),
        })

    try:
        checkpoint(0)
        completed_updates = 0
        next_sim_checkpoint = eval_sim_step_interval
        while True:
            if (training_sim_step_budget is not None
                    and train_env.training_sim_steps >= training_sim_step_budget):
                break
            if training_sim_step_budget is None and completed_updates >= steps:
                break
            if completed_updates >= steps:
                raise RuntimeError(
                    "training hit the frozen update safety cap before the "
                    "simulator-step budget")
            completed_updates += 1
            started = time.perf_counter()
            stats = trainer.step()
            totals["training_wall_seconds"] += time.perf_counter() - started
            totals["live_groups"] += stats.live_groups
            totals["dead_groups"] += stats.dead_groups
            totals["relabeled_groups"] += stats.relabeled_groups
            if training_sim_step_budget is None:
                should_checkpoint = (
                    completed_updates % eval_every == 0
                    or completed_updates == steps)
            else:
                should_checkpoint = (
                    train_env.training_sim_steps >= next_sim_checkpoint
                    or train_env.training_sim_steps >= training_sim_step_budget)
            if should_checkpoint:
                checkpoint(completed_updates)
                if next_sim_checkpoint is not None:
                    while next_sim_checkpoint <= train_env.training_sim_steps:
                        next_sim_checkpoint += eval_sim_step_interval
        if history[-1]["step"] != completed_updates:
            checkpoint(completed_updates)
    finally:
        _close_if_supported(eval_env)
        _close_if_supported(train_env)

    return {
        "arm": arm,
        "seed": int(seed),
        "teacher_seed": teacher_seed,
        "eval_seed": eval_seed,
        "target_uniform_auc_by_episode": _normalized_auc(history, "episodes"),
        "target_uniform_auc_by_sim_step": _normalized_auc(
            history, "sim_steps", budget=training_sim_step_budget),
        "target_uniform_auc_by_own_training_wall": _normalized_auc(
            history, "training_wall_seconds"),
        "final": history[-1],
        "history": history,
        "training_episode_records": instrumented.training_episode_records,
    }


def run_campaign(
    *,
    dataset_root: Path,
    split_path: Path,
    seed: int,
    manifest_path: Path = DEFAULT_MANIFEST,
    robot_sdf: Path = DEFAULT_ROBOT_SDF,
    stepper_path: Path | None = None,
    prereg_path: Path = DEFAULT_PREREG,
    analyzer_path: Path = DEFAULT_ANALYZER,
    protocol_path: Path = DEFAULT_PROTOCOL,
    campaign_cell: str = "primary",
    runtime_root: Path = Path("icra2027/results/barn_runtime"),
    arms: Sequence[str] = ARM_NAMES,
    steps: int = 60,
    n_rollouts: int = 16,
    tasks_per_step: int = 4,
    eval_every: int = 10,
    eval_episodes: int = 8,
    episode_timeout: float = 25.0,
    max_step_size: float = 0.005,
    real_time_update_rate: int = 2000,
    domain_id: int | None = None,
    master_port: int | None = None,
    teacher_floor: float = 0.1,
    teacher_decay: float = 0.7,
    teacher_gamma: float = 1.0,
    staged_initial_strata: int = 1,
    staged_promotion_threshold: float = 0.7,
    staged_min_frontier_groups: int = 5,
    training_sim_step_budget: int | None = None,
    eval_sim_step_interval: int | None = None,
    smoke: bool = False,
    engineering_course_id: str | None = None,
    verify_assets: bool = True,
    expected_manifest_sha256: str | None = None,
    expected_split_sha256: str | None = None,
    expected_prereg_sha256: str | None = None,
    expected_analyzer_sha256: str | None = None,
    expected_protocol_sha256: str | None = None,
    container_sha256: str | None = None,
    source_sha256: str | None = None,
    campaign_id: str | None = None,
    attempt_id: str | None = None,
    submitted_utc: str | None = None,
    slurm_job_id: str | None = None,
    slurm_array_job_id: str | None = None,
    slurm_array_task_id: int | None = None,
) -> dict:
    """Run all four arms for one paired seed and return one analyzer artifact."""

    arms = tuple(arms)
    unknown = set(arms) - set(ARM_NAMES)
    if unknown or len(arms) != len(set(arms)):
        raise ValueError(f"invalid or duplicate arms: {list(arms)!r}")
    if not smoke:
        expected_arms = (ARM_NAMES if campaign_cell == "primary" else
                         ("ours_uN", "learnability"))
        if arms != expected_arms:
            raise ValueError(
                f"non-smoke {campaign_cell} runs require all four arms or "
                f"the exact frozen ablation arms: "
                f"{list(expected_arms)!r}")
    if engineering_course_id is not None and not smoke:
        raise ValueError(
            "engineering_course_id may be used only for non-evidentiary smoke runs")
    if smoke and engineering_course_id is None:
        raise ValueError(
            "real BARN smoke runs require an explicit frozen-training-partition "
            "engineering_course_id; held-out smoke evaluation is forbidden")
    if not verify_assets and not smoke:
        raise ValueError("asset checksum verification may be skipped only in smoke mode")
    if not smoke and any(value is None for value in (
            expected_manifest_sha256, expected_split_sha256,
            expected_prereg_sha256, expected_analyzer_sha256,
            expected_protocol_sha256, container_sha256, source_sha256)):
        raise ValueError(
            "evidence runs require expected manifest, split, preregistration, "
            "analyzer, protocol, container, and source hashes")
    if not smoke and (training_sim_step_budget is None
                      or eval_sim_step_interval is None):
        raise ValueError(
            "evidence runs require a frozen simulator-step budget and "
            "simulator-step evaluation interval")
    if seed < 0:
        raise ValueError("campaign seed must be non-negative")
    execution = (None if smoke else _execution_identity(
        campaign_id=campaign_id,
        attempt_id=attempt_id,
        submitted_utc=submitted_utc,
        slurm_job_id=slurm_job_id,
        slurm_array_job_id=slurm_array_job_id,
        slurm_array_task_id=slurm_array_task_id,
    ))
    if execution is not None and execution["slurm_array_task_id"] != seed:
        raise ValueError("slurm_array_task_id must equal the campaign seed")

    dataset_root = Path(dataset_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    split_path = Path(split_path).resolve()
    robot_sdf = Path(robot_sdf).resolve()
    stepper_path = (Path(stepper_path).resolve()
                    if stepper_path is not None else None)
    prereg_path = Path(prereg_path).resolve()
    analyzer_path = Path(analyzer_path).resolve()
    protocol_path = Path(protocol_path).resolve()
    runtime_root = Path(runtime_root).resolve()
    if not robot_sdf.is_file():
        raise FileNotFoundError(robot_sdf)
    if stepper_path is not None:
        if not stepper_path.is_file():
            raise FileNotFoundError(stepper_path)
        exact_step_plugin = stepper_path.parent / "libbarn_exact_step.so"
        if not exact_step_plugin.is_file():
            raise FileNotFoundError(exact_step_plugin)
    else:
        exact_step_plugin = None
    split, manifest_rows = load_frozen_inputs(manifest_path, split_path)
    manifest_sha = _sha256(manifest_path)
    split_sha = _sha256(split_path)
    prereg_sha = (_verify_frozen_prereg(
        prereg_path, expected_prereg_sha256)
        if not smoke else (_sha256(prereg_path) if prereg_path.is_file() else None))
    if not smoke:
        protocol_sha, protocol, contract = _verify_frozen_protocol(
            protocol_path, expected_protocol_sha256, campaign_cell)
    else:
        protocol_sha = (_sha256(protocol_path)
                        if protocol_path.is_file() else None)
        protocol, contract = None, None
    if not analyzer_path.is_file():
        raise FileNotFoundError(analyzer_path)
    analyzer_sha = _sha256(analyzer_path)
    if expected_analyzer_sha256 is not None:
        expected = _require_sha256(
            expected_analyzer_sha256, field="expected_analyzer_sha256")
        if not hmac.compare_digest(analyzer_sha, expected):
            raise ValueError(
                f"analyzer hash mismatch: expected {expected}, got {analyzer_sha}")
    pinned_container_sha = (_require_sha256(
        container_sha256, field="container_sha256")
        if container_sha256 is not None else None)
    pinned_source_sha = (_require_sha256(
        source_sha256, field="source_sha256")
        if source_sha256 is not None else None)
    if expected_manifest_sha256 is not None:
        expected = _require_sha256(
            expected_manifest_sha256, field="expected_manifest_sha256")
        if not hmac.compare_digest(manifest_sha, expected):
            raise ValueError(
                f"manifest hash mismatch: expected {expected}, got {manifest_sha}")
    if expected_split_sha256 is not None:
        expected = _require_sha256(
            expected_split_sha256, field="expected_split_sha256")
        if not hmac.compare_digest(split_sha, expected):
            raise ValueError(
                f"split hash mismatch: expected {expected}, got {split_sha}")
    if not smoke:
        _enforce_protocol_contract(contract, {
            "campaign_cell": campaign_cell,
            "protocol_id": protocol.get("protocol_id"),
            "evidence_status": contract["evidence_status"],
            "arms": list(arms),
            "execution_order_by_seed": contract["execution_order_by_seed"],
            "n_rollouts": n_rollouts,
            "seeds": protocol["shared_training"]["seeds"],
            "tasks_per_step": tasks_per_step,
            "eval_episodes": eval_episodes,
            "training_sim_step_budget": training_sim_step_budget,
            "eval_sim_step_interval": eval_sim_step_interval,
            "steps": steps,
            "eval_every": eval_every,
            "teacher_floor": teacher_floor,
            "teacher_decay": teacher_decay,
            "teacher_gamma": teacher_gamma,
            "staged_initial_strata": staged_initial_strata,
            "staged_promotion_threshold": staged_promotion_threshold,
            "staged_min_frontier_groups": staged_min_frontier_groups,
            "episode_timeout": episode_timeout,
            "max_step_size": max_step_size,
            "real_time_update_rate": real_time_update_rate,
        })
        if seed not in contract["seeds"]:
            raise ValueError(
                f"seed {seed} is not in frozen seed list {contract['seeds']}")
        dataset_contract = protocol["dataset"]
        environment_contract = protocol["environment"]
        pinned_dataset_hashes = {
            "manifest_sha256": manifest_sha,
            "split_sha256": split_sha,
        }
        for field, actual in pinned_dataset_hashes.items():
            expected = _require_sha256(
                dataset_contract.get(field), field=f"protocol dataset {field}")
            if not hmac.compare_digest(actual, expected):
                raise ValueError(
                    f"{field} differs from machine-readable protocol")
        protocol_container = _require_sha256(
            environment_contract.get("container_sha256"),
            field="protocol environment container_sha256")
        if not hmac.compare_digest(pinned_container_sha, protocol_container):
            raise ValueError("container hash differs from machine-readable protocol")
        protocol_analyzer = _require_sha256(
            protocol.get("analysis", {}).get("analyzer_sha256"),
            field="protocol analysis analyzer_sha256")
        if not hmac.compare_digest(analyzer_sha, protocol_analyzer):
            raise ValueError("analyzer hash differs from machine-readable protocol")
        split_contract = {
            "seed": dataset_contract.get("split_seed"),
            "n_strata": dataset_contract.get("n_strata"),
            "n_train": dataset_contract.get("n_train_courses"),
            "n_heldout": dataset_contract.get("n_heldout_courses"),
        }
        for field, expected in split_contract.items():
            if split.get(field) != expected:
                raise ValueError(
                    f"split {field}={split.get(field)!r} differs from frozen "
                    f"protocol value {expected!r}")
    n_strata = int(split["n_strata"])
    if engineering_course_id is not None:
        if engineering_course_id not in split["train_ids"]:
            raise ValueError(
                "engineering smoke course must belong to the frozen training "
                f"partition: {engineering_course_id!r}")
        engineering_row = next(
            row for row in manifest_rows
            if row["env_id"] == engineering_course_id)
        if verify_assets:
            # A timing smoke must remain blind to the prospective held-out
            # filesystem.  Verify only the explicitly selected frozen-train
            # course; never resolve, stat, hash, or load a held-out asset.
            verify_dataset_assets(dataset_root, [engineering_row])
        engineering_course = load_courses(
            manifest_path, dataset_root, [engineering_course_id])[0]
        # A throughput smoke must exercise the real training/update/evaluation
        # plumbing without exposing a prospective held-out course.  Reuse the
        # selected training course for the smoke-only evaluation adapter and
        # make the non-evidentiary override explicit in the artifact.
        train_courses = [engineering_course]
        heldout_courses = [engineering_course]
        n_strata = 1
    else:
        if verify_assets:
            verify_dataset_assets(dataset_root, manifest_rows)
        train_courses = load_courses(
            manifest_path, dataset_root, split["train_ids"])
        heldout_courses = load_courses(
            manifest_path, dataset_root, split["heldout_ids"])
    resolved_domain_id = (int(os.environ["ROS_DOMAIN_ID"])
                          if domain_id is None and "ROS_DOMAIN_ID" in os.environ
                          else (80 + 2 * (seed % 50)
                                if domain_id is None else int(domain_id)))
    resolved_master_port = (11500 + 4 * (seed % 100)
                            if master_port is None else int(master_port))
    if not smoke:
        isolation = protocol.get("isolation")
        if not isinstance(isolation, dict):
            raise ValueError("protocol is missing cross-cell isolation settings")
        try:
            expected_domain_id = (
                int(isolation["domain_base_by_cell"][campaign_cell])
                + seed * int(isolation["seed_stride"]))
            expected_master_port = (
                int(isolation["master_port_base_by_cell"][campaign_cell])
                + seed * int(isolation["master_port_seed_stride"]))
            eval_domain_offset = int(isolation["eval_offset"])
            eval_port_offset = int(isolation["eval_master_port_offset"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"protocol has invalid isolation mapping for {campaign_cell}") from error
        if eval_domain_offset != 1 or eval_port_offset != 1:
            raise ValueError("runner requires frozen evaluation isolation offsets of 1")
        if resolved_domain_id != expected_domain_id:
            raise ValueError(
                f"domain_id={resolved_domain_id} differs from frozen protocol "
                f"value {expected_domain_id}")
        if resolved_master_port != expected_master_port:
            raise ValueError(
                f"master_port={resolved_master_port} differs from frozen protocol "
                f"value {expected_master_port}")
        if not 0 <= resolved_domain_id < 232:
            raise ValueError("frozen ROS domain pair is out of range")
        if not 1024 <= resolved_master_port < 65535:
            raise ValueError("frozen Gazebo master port pair is out of range")

    if smoke:
        execution_order = arms
    else:
        order = contract["execution_order_by_seed"].get(str(seed))
        if (not isinstance(order, list) or len(order) != len(arms)
                or set(order) != set(arms) or len(order) != len(set(order))):
            raise ValueError(
                f"protocol has invalid execution order for seed {seed}: {order!r}")
        execution_order = tuple(order)

    results = {arm: [] for arm in arms}
    for arm in execution_order:
        results[arm].append(run_one(
            arm,
            seed,
            train_courses=train_courses,
            heldout_courses=heldout_courses,
            n_strata=n_strata,
            robot_sdf=robot_sdf,
            stepper_path=stepper_path,
            runtime_root=runtime_root,
            steps=steps,
            n_rollouts=n_rollouts,
            tasks_per_step=tasks_per_step,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            episode_timeout=episode_timeout,
            max_step_size=max_step_size,
            real_time_update_rate=real_time_update_rate,
            domain_id=resolved_domain_id,
            master_port=resolved_master_port,
            teacher_floor=teacher_floor,
            teacher_decay=teacher_decay,
            teacher_gamma=teacher_gamma,
            staged_initial_strata=staged_initial_strata,
            staged_promotion_threshold=staged_promotion_threshold,
            staged_min_frontier_groups=staged_min_frontier_groups,
            training_sim_step_budget=training_sim_step_budget,
            eval_sim_step_interval=eval_sim_step_interval,
        ))

    train_strata = _strata(train_courses, n_strata)
    heldout_strata = _strata(heldout_courses, n_strata)
    artifact = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": (
            SMOKE_EVIDENCE_STATUS if smoke else contract["evidence_status"]),
        "domain": "barn_gazebo_cpu_navigation",
        "execution": execution,
        "heldout_protocol": (
            ("engineering smoke reuses one frozen-training-partition course "
             "in the isolated evaluation adapter; no held-out course is read")
            if engineering_course_id is not None else
            ("fixed course-level seeds shared across arms and checkpoints; "
             "a separate held-out BarnGazeboSpace shares only the policy and "
             "cannot mutate training RNG, teacher, launch, or budget state")),
        "provenance": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "split_path": str(split_path),
            "split_sha256": split_sha,
            "split_bound_manifest_sha256": split["source_sha256"],
            "prereg_path": str(prereg_path),
            "prereg_sha256": prereg_sha,
            "analyzer_path": str(analyzer_path),
            "analyzer_sha256": analyzer_sha,
            "protocol_path": str(protocol_path),
            "protocol_sha256": protocol_sha,
            "container_sha256": pinned_container_sha,
            "source_sha256": pinned_source_sha,
            "dataset_root": str(dataset_root),
            "robot_sdf": str(robot_sdf),
            "robot_sdf_sha256": _sha256(robot_sdf),
            "asset_hashes_verified": bool(verify_assets),
        },
        "config": {
            "arms": list(arms),
            "execution_order": list(execution_order),
            "campaign_cell": campaign_cell,
            "protocol_id": (protocol.get("protocol_id")
                            if protocol is not None else None),
            "seeds": 1,
            "seed_start": int(seed),
            "seed_list": [int(seed)],
            "steps": steps,
            "n_rollouts": n_rollouts,
            "tasks_per_step": tasks_per_step,
            "eval_every": eval_every,
            "eval_episodes": eval_episodes,
            "training_sim_step_budget": training_sim_step_budget,
            "eval_sim_step_interval": eval_sim_step_interval,
            "max_training_updates": steps,
            "episode_timeout": episode_timeout,
            "max_step_size": max_step_size,
            "real_time_update_rate": real_time_update_rate,
            "hindsight": False,
            "estimator": "maxrl",
            "teacher_gamma": teacher_gamma,
            "teacher_decay": teacher_decay,
            "teacher_floor": teacher_floor,
            "teacher_unit": "frozen_difficulty_stratum",
            "n_strata": n_strata,
            "difficulty_metadata": (
                "published optimal traversal time seconds; longer is harder"),
            "staged_initial_strata": staged_initial_strata,
            "staged_promotion_threshold": staged_promotion_threshold,
            "staged_min_frontier_groups": staged_min_frontier_groups,
            "n_train_courses": len(train_courses),
            "n_heldout_courses": len(heldout_courses),
            "train_strata": [{
                "stratum": index,
                "n_courses": len(courses),
                "difficulty_min": float(courses[0].difficulty),
                "difficulty_max": float(courses[-1].difficulty),
                "course_ids": [course.env_id for course in courses],
            } for index, courses in enumerate(train_strata)],
            "heldout_strata": [{
                "stratum": index,
                "n_courses": len(courses),
                "difficulty_min": float(courses[0].difficulty),
                "difficulty_max": float(courses[-1].difficulty),
                "course_ids": [course.env_id for course in courses],
            } for index, courses in enumerate(heldout_strata)],
            "campaign_seed": int(seed),
            "split_seed": int(split["seed"]),
            "domain_id": resolved_domain_id,
            "eval_domain_id": resolved_domain_id + 1,
            "master_port": resolved_master_port,
            "eval_master_port": resolved_master_port + 1,
            "runtime_root": str(runtime_root),
            "smoke": bool(smoke),
            "engineering_course_id": engineering_course_id,
            "evaluation_partition": (
                "training_course_engineering_smoke"
                if engineering_course_id is not None else "frozen_heldout"),
        },
        "results": results,
    }
    if stepper_path is not None:
        artifact["provenance"].update({
            "gazebo_stepper_sha256": _sha256(stepper_path),
            "exact_step_plugin_sha256": _sha256(exact_step_plugin),
        })
    if artifact["evidence_status"] in {
            FULL_EVIDENCE_STATUS, ABLATION_EVIDENCE_STATUS} and smoke:
        raise AssertionError("smoke artifact cannot be marked as full evidence")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one paired seed of the four-arm CPU BARN campaign")
    # --backend and --out retain compatibility with the Hopper template; this
    # module has only one backend and --output is the canonical spelling.
    parser.add_argument("--backend", choices=("barn_gazebo",),
                        default="barn_gazebo")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", dest="split_path", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", "--out", dest="output", type=Path,
                        required=True)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--robot-sdf", type=Path, default=DEFAULT_ROBOT_SDF)
    parser.add_argument(
        "--stepper-path", type=Path,
        help="compiled gazebo_stepper; matching plugin must be in its directory")
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--analyzer", type=Path, default=DEFAULT_ANALYZER)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--campaign-cell", default="primary",
        choices=("primary", "ablation_n2", "ablation_n4", "ablation_n16"))
    parser.add_argument("--arms", default=",".join(ARM_NAMES))
    parser.add_argument("--steps", type=int, default=200,
                        help="hard safety cap on trainer updates")
    parser.add_argument("--n-rollouts", type=int, default=8)
    parser.add_argument("--tasks-per-step", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--training-sim-step-budget", type=int,
                        default=1_000_000)
    parser.add_argument("--eval-sim-step-interval", type=int,
                        default=200_000)
    parser.add_argument("--episode-timeout", type=float, default=25.0)
    parser.add_argument("--max-step-size", type=float, default=0.005)
    parser.add_argument("--real-time-update-rate", type=int, default=2000)
    parser.add_argument("--domain-id", type=int)
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--teacher-floor", type=float, default=0.1)
    parser.add_argument("--teacher-decay", type=float, default=0.7)
    parser.add_argument("--teacher-gamma", type=float, default=1.0)
    parser.add_argument("--staged-initial-strata", type=int, default=1)
    parser.add_argument("--staged-promotion-threshold", type=float,
                        default=0.7)
    parser.add_argument("--staged-min-frontier-groups", type=int, default=5)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-split-sha256")
    parser.add_argument("--expected-prereg-sha256")
    parser.add_argument("--expected-analyzer-sha256")
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--container-sha256")
    parser.add_argument("--source-sha256")
    parser.add_argument("--campaign-id")
    parser.add_argument("--attempt-id")
    parser.add_argument("--submitted-utc")
    parser.add_argument("--slurm-job-id")
    parser.add_argument("--slurm-array-job-id")
    parser.add_argument("--slurm-array-task-id", type=int)
    parser.add_argument(
        "--engineering-course-id",
        help=("smoke-only training-partition course used for both training and "
              "isolated plumbing evaluation; never reads held-out courses"))
    parser.add_argument(
        "--smoke-update-budget", action="store_true",
        help=("smoke-only: use --steps as the update budget instead of the "
              "full evidence simulator-step budget"))
    parser.add_argument(
        "--smoke", action="store_true",
        help="mark this real-backend plumbing run as non-evidentiary")
    parser.add_argument(
        "--skip-asset-hash-check", action="store_true",
        help="smoke-only shortcut; full runs always verify bound asset hashes")
    return parser


def _atomic_write_json(path: Path, artifact: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite campaign artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x") as handle:
            json.dump(artifact, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and refuses an existing destination,
        # unlike os.replace, so a completed attempt can never be overwritten.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    arms = tuple(part.strip() for part in args.arms.split(",") if part.strip())
    unknown = set(arms) - set(ARM_NAMES)
    if unknown:
        parser.error(f"unknown arms: {sorted(unknown)}")
    if args.engineering_course_id is not None and not args.smoke:
        parser.error("--engineering-course-id requires --smoke")
    if args.smoke_update_budget and not args.smoke:
        parser.error("--smoke-update-budget requires --smoke")
    runtime_root = args.runtime_root or args.output.with_name(
        args.output.name + "_runtime")
    artifact = run_campaign(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        split_path=args.split_path,
        seed=args.seed,
        robot_sdf=args.robot_sdf,
        stepper_path=args.stepper_path,
        prereg_path=args.prereg,
        analyzer_path=args.analyzer,
        protocol_path=args.protocol,
        campaign_cell=args.campaign_cell,
        runtime_root=runtime_root,
        arms=arms,
        steps=args.steps,
        n_rollouts=args.n_rollouts,
        tasks_per_step=args.tasks_per_step,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        episode_timeout=args.episode_timeout,
        max_step_size=args.max_step_size,
        real_time_update_rate=args.real_time_update_rate,
        domain_id=args.domain_id,
        master_port=args.master_port,
        teacher_floor=args.teacher_floor,
        teacher_decay=args.teacher_decay,
        teacher_gamma=args.teacher_gamma,
        staged_initial_strata=args.staged_initial_strata,
        staged_promotion_threshold=args.staged_promotion_threshold,
        staged_min_frontier_groups=args.staged_min_frontier_groups,
        training_sim_step_budget=(
            None if args.smoke_update_budget
            else args.training_sim_step_budget),
        eval_sim_step_interval=(
            None if args.smoke_update_budget
            else args.eval_sim_step_interval),
        smoke=args.smoke,
        engineering_course_id=args.engineering_course_id,
        verify_assets=not args.skip_asset_hash_check,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_split_sha256=args.expected_split_sha256,
        expected_prereg_sha256=args.expected_prereg_sha256,
        expected_analyzer_sha256=args.expected_analyzer_sha256,
        expected_protocol_sha256=args.expected_protocol_sha256,
        container_sha256=args.container_sha256,
        source_sha256=args.source_sha256,
        campaign_id=args.campaign_id,
        attempt_id=args.attempt_id,
        submitted_utc=args.submitted_utc,
        slurm_job_id=args.slurm_job_id,
        slurm_array_job_id=args.slurm_array_job_id,
        slurm_array_task_id=args.slurm_array_task_id,
    )
    _atomic_write_json(args.output, artifact)
    # Do not print partial or endpoint performance: the preregistration bars
    # outcome peeking before the complete paired matrix is analyzed.
    print(
        f"wrote {args.output}: seed={args.seed} arms={len(arms)} "
        f"status={artifact['evidence_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
