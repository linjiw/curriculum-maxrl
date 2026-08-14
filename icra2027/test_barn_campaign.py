"""Focused CPU-only tests for :mod:`icra2027.barn_campaign`.

The fake adapter implements the same trainer-facing surface as
``BarnGazeboSpace`` but never imports ROS or starts Gazebo.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from frontier_rl.interfaces import GroupResult
from icra2027 import barn_campaign
from icra2027.freeze_pool_split import make_manifest


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _frozen_fixture(tmp_path: Path, *, n_courses: int = 8):
    dataset = tmp_path / "dataset"
    worlds = dataset / "world_files"
    paths = dataset / "path_files"
    grids = dataset / "grid_files"
    worlds.mkdir(parents=True)
    paths.mkdir(parents=True)
    grids.mkdir(parents=True)
    records = []
    for index in range(n_courses):
        world_data = f"fake-world-{index}\n".encode()
        path_data = f"fake-path-{index}\n".encode()
        grid_data = f"fake-grid-{index}\n".encode()
        world = worlds / f"world_{index}.world"
        path = paths / f"path_{index}.npy"
        grid = grids / f"grid_{index}.npy"
        world.write_bytes(world_data)
        path.write_bytes(path_data)
        grid.write_bytes(grid_data)
        records.append({
            "env_id": f"barn-{index:03d}",
            "barn_index": index,
            "difficulty": float(index + 1),
            "asset": f"world_files/{world.name}",
            "asset_sha256": _digest(world_data),
            "path_asset": f"path_files/{path.name}",
            "path_sha256": _digest(path_data),
            "grid_asset": f"grid_files/{grid.name}",
            "grid_sha256": _digest(grid_data),
        })
    manifest = tmp_path / "barn_manifest.jsonl"
    manifest.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in records))
    split = tmp_path / "barn_split.json"
    split.write_text(json.dumps(make_manifest(
        manifest, holdout_fraction=0.25, n_strata=2, seed=20270811),
        indent=2) + "\n")
    robot = tmp_path / "barn_diff_drive.sdf"
    robot.write_text("<sdf version='1.6'/>\n")
    return dataset, manifest, split, robot


def _frozen_protocol(
    tmp_path: Path,
    manifest: Path,
    split: Path,
    *,
    steps: int = 2,
    n_rollouts: int = 2,
    tasks_per_step: int = 2,
    eval_every: int = 1,
    eval_episodes: int = 2,
    training_sim_step_budget: int = 56,
    eval_sim_step_interval: int = 28,
) -> Path:
    frozen_split = json.loads(split.read_text())
    protocol = {
        "schema_version": 1,
        "status": "FROZEN",
        "protocol_id": "test-barn-v1",
        "domain": "barn_gazebo_cpu_navigation",
        "dataset": {
            "manifest_sha256": barn_campaign._sha256(manifest),
            "split_sha256": barn_campaign._sha256(split),
            "split_seed": frozen_split["seed"],
            "n_strata": frozen_split["n_strata"],
            "n_train_courses": frozen_split["n_train"],
            "n_heldout_courses": frozen_split["n_heldout"],
        },
        "environment": {
            "container_sha256": "a" * 64,
            "cpu_only": True,
            "episode_timeout": 25.0,
            "max_step_size": 0.005,
            "real_time_update_rate": 2000,
        },
        "analysis": {
            "analyzer_sha256": barn_campaign._sha256(
                barn_campaign.DEFAULT_ANALYZER),
        },
        "shared_training": {
            "seeds": [3],
            "tasks_per_step": tasks_per_step,
            "eval_episodes": eval_episodes,
            "training_sim_step_budget": training_sim_step_budget,
            "eval_sim_step_interval": eval_sim_step_interval,
            "max_training_updates": steps,
            "eval_every": eval_every,
            "teacher_floor": 0.1,
            "teacher_decay": 0.7,
            "teacher_gamma": 1.0,
            "staged_initial_strata": 1,
            "staged_promotion_threshold": 0.7,
            "staged_min_frontier_groups": 5,
        },
        "primary": {
            "evidence_status": "full_barn_campaign",
            "arms": list(barn_campaign.ARM_NAMES),
            "n_rollouts": n_rollouts,
            "execution_order_by_seed": {
                "3": list(barn_campaign.ARM_NAMES)},
        },
        "ablation": {
            "evidence_status": "full_barn_n_ablation",
            "arms": ["ours_uN", "learnability"],
            "n_values": [2, 4, 8, 16],
            "fresh_cell_names": [
                "ablation_n2", "ablation_n4", "ablation_n16"],
            "execution_order_by_seed": {
                "3": ["ours_uN", "learnability"]},
        },
        "isolation": {
            "seed_stride": 2,
            "eval_offset": 1,
            "domain_base_by_cell": {
                "primary": 20,
                "ablation_n2": 50,
                "ablation_n4": 80,
                "ablation_n16": 110,
            },
            "master_port_seed_stride": 4,
            "eval_master_port_offset": 1,
            "master_port_base_by_cell": {
                "primary": 13000,
                "ablation_n2": 14000,
                "ablation_n4": 15000,
                "ablation_n16": 16000,
            },
        },
    }
    path = tmp_path / "barn_protocol.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    return path


class _FakePolicy:
    def __init__(self):
        self.weight = np.zeros(1, dtype=float)
        self.updates = 0

    def state_dict(self):
        return {"weight": self.weight.copy()}


class _FakeBarnGazeboSpace:
    instances = []
    eval_calls = []

    @classmethod
    def reset(cls):
        cls.instances = []
        cls.eval_calls = []

    def __init__(self, courses, robot_sdf, runtime_root, *, seed,
                 n_strata, domain_id, master_port, episode_timeout,
                 max_step_size, real_time_update_rate, policy=None,
                 stepper_path=None):
        self.courses = list(courses)
        self.robot_sdf = Path(robot_sdf)
        self.stepper_path = (Path(stepper_path) if stepper_path is not None
                             else None)
        self.runtime_root = Path(runtime_root)
        self.seed = int(seed)
        self.n_strata = int(n_strata)
        self.domain_id = int(domain_id)
        self.master_port = int(master_port)
        self.policy = policy or _FakePolicy()
        self.rng = np.random.default_rng(seed)
        self.training_episodes = 0
        self.training_sim_steps = 0
        self.course_launches = 0
        self._episode_counter = 0
        self.closed = False
        type(self).instances.append(self)

    @property
    def n_tasks(self):
        return self.n_strata

    def rollout_group(self, task_id, n_rollouts):
        assert 0 <= int(task_id) < self.n_strata
        # Touch only the training adapter RNG, as the real adapter does.
        self.rng.integers(0, 2**31 - 1)
        rewards = np.asarray(
            [(episode + int(task_id)) % 2 for episode in range(n_rollouts)],
            dtype=float)
        self.training_episodes += n_rollouts
        self.training_sim_steps += 7 * n_rollouts
        self.course_launches += 1
        self._episode_counter += n_rollouts
        return GroupResult(
            int(task_id), rewards,
            trajectories=[{"episode": index} for index in range(n_rollouts)],
            infos=[{"sim_steps": 7} for _ in range(n_rollouts)],
        )

    def relabel(self, group):
        return None

    def update(self, task_id, trajectories, weights):
        self.policy.weight += 0.01 * (int(task_id) + 1)
        self.policy.updates += 1

    def evaluate_course(self, course, n, *, seed):
        # Evaluation mutates its *own* operational launch state.  If the
        # runner accidentally evaluates through the training adapter, the
        # state-isolation assertion will detect it.
        self.course_launches += 1
        type(self).eval_calls.append({
            "adapter": id(self),
            "course": course.env_id,
            "seed": int(seed),
            "policy": id(self.policy),
        })
        successes = (course.barn_index + self.policy.updates) % (n + 1)
        episodes = [
            {"status": "succeeded" if index < successes else "timeout"}
            for index in range(n)
        ]
        return successes, 11 * n, episodes

    def close(self):
        self.closed = True


def _run_fake(monkeypatch, tmp_path, *, smoke=True, arms=barn_campaign.ARM_NAMES):
    dataset, manifest, split, robot = _frozen_fixture(tmp_path)
    prereg = tmp_path / "prereg_icra.md"
    prereg.write_text("# Test preregistration\n\n**Status:** FROZEN\n")
    _FakeBarnGazeboSpace.reset()
    monkeypatch.setattr(
        barn_campaign, "BarnGazeboSpace", _FakeBarnGazeboSpace)
    protocol = _frozen_protocol(tmp_path, manifest, split)
    frozen_split = json.loads(split.read_text())
    artifact = barn_campaign.run_campaign(
        dataset_root=dataset,
        manifest_path=manifest,
        split_path=split,
        prereg_path=prereg,
        robot_sdf=robot,
        runtime_root=tmp_path / "runtime",
        seed=3,
        arms=arms,
        steps=2,
        n_rollouts=2,
        tasks_per_step=2,
        eval_every=1,
        eval_episodes=2,
        domain_id=26,
        master_port=13012,
        training_sim_step_budget=56,
        eval_sim_step_interval=28,
        smoke=smoke,
        engineering_course_id=(
            frozen_split["train_ids"][0] if smoke else None),
        expected_manifest_sha256=barn_campaign._sha256(manifest),
        expected_split_sha256=barn_campaign._sha256(split),
        expected_prereg_sha256=barn_campaign._sha256(prereg),
        expected_analyzer_sha256=barn_campaign._sha256(
            barn_campaign.DEFAULT_ANALYZER),
        protocol_path=protocol,
        expected_protocol_sha256=barn_campaign._sha256(protocol),
        container_sha256="a" * 64,
        source_sha256="b" * 64,
        campaign_id="test-primary",
        attempt_id="attempt-001",
        submitted_utc="2026-08-14T00:00:00Z",
        slurm_job_id="12345_3",
        slurm_array_job_id="12345",
        slurm_array_task_id=3,
    )
    return artifact


def test_frozen_split_is_bound_to_manifest_and_rederived(tmp_path):
    _, manifest, split, _ = _frozen_fixture(tmp_path)
    loaded, rows = barn_campaign.load_frozen_inputs(manifest, split)
    assert loaded["n_total"] == len(rows) == 8

    bad_hash = tmp_path / "bad_hash.json"
    payload = json.loads(split.read_text())
    payload["source_sha256"] = "0" * 64
    bad_hash.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="split/manifest hash mismatch"):
        barn_campaign.load_frozen_inputs(manifest, bad_hash)

    bad_ids = tmp_path / "bad_ids.json"
    payload = json.loads(split.read_text())
    payload["train_ids"][0], payload["heldout_ids"][0] = (
        payload["heldout_ids"][0], payload["train_ids"][0])
    bad_ids.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="do not reproduce"):
        barn_campaign.load_frozen_inputs(manifest, bad_ids)


def test_asset_hashes_fail_closed(tmp_path):
    dataset, manifest, split, _ = _frozen_fixture(tmp_path)
    frozen, rows = barn_campaign.load_frozen_inputs(manifest, split)
    assert frozen["source_sha256"] == barn_campaign._sha256(manifest)
    barn_campaign.verify_dataset_assets(dataset, rows)
    (dataset / rows[0]["asset"]).write_text("tampered")
    with pytest.raises(ValueError, match="asset hash mismatch"):
        barn_campaign.verify_dataset_assets(dataset, rows)


def test_four_arm_artifact_is_paired_isolated_and_analyzer_compatible(
        monkeypatch, tmp_path):
    artifact = _run_fake(monkeypatch, tmp_path, smoke=True)
    assert artifact["evidence_status"] == (
        "engineering_smoke_not_paper_evidence")
    assert artifact["config"]["teacher_unit"] == "frozen_difficulty_stratum"
    assert set(artifact["results"]) == set(barn_campaign.ARM_NAMES)

    # Two adapter instances per arm: train and held out.  Both share a policy,
    # but their RNGs and budget counters are distinct.
    assert len(_FakeBarnGazeboSpace.instances) == 8
    for offset in range(0, 8, 2):
        train, heldout = _FakeBarnGazeboSpace.instances[offset:offset + 2]
        assert train.policy is heldout.policy
        assert train.seed == 3
        assert train.training_episodes == 8
        assert train.training_sim_steps == 56
        assert heldout.training_episodes == 0
        assert heldout.training_sim_steps == 0
        assert train.closed and heldout.closed

    # Every course reuses one seed across checkpoints and arms.
    seeds_by_course = {}
    for call in _FakeBarnGazeboSpace.eval_calls:
        seeds_by_course.setdefault(call["course"], set()).add(call["seed"])
    assert seeds_by_course and all(len(seeds) == 1
                                   for seeds in seeds_by_course.values())

    for arm, runs in artifact["results"].items():
        run = runs[0]
        assert run["seed"] == 3
        assert run["final"]["episodes"] == 8
        assert run["final"]["sim_steps"] == 56
        assert len(run["final"]["teacher"]["posterior_mean"]) == 1
        assert "target_uniform_auc_by_sim_step" in run
        assert len(run["history"]) == 3

    # Production analysis is deliberately stricter than runner plumbing: it
    # accepts only a five-seed, protocol-bound merged evidence artifact.
    assert artifact["evidence_status"] != "full_barn_campaign"


def test_full_status_requires_non_smoke_complete_four_arm_run(
        monkeypatch, tmp_path):
    artifact = _run_fake(monkeypatch, tmp_path, smoke=False)
    assert artifact["evidence_status"] == "full_barn_campaign"
    with pytest.raises(ValueError, match="all four arms"):
        barn_campaign.run_campaign(
            dataset_root=tmp_path / "dataset",
            manifest_path=tmp_path / "barn_manifest.jsonl",
            split_path=tmp_path / "barn_split.json",
            robot_sdf=tmp_path / "barn_diff_drive.sdf",
            runtime_root=tmp_path / "runtime2",
            seed=3,
            arms=("ours_uN",),
            smoke=False,
        )


def test_full_status_requires_pinned_frozen_prereg(monkeypatch, tmp_path):
    dataset, manifest, split, robot = _frozen_fixture(tmp_path)
    prereg = tmp_path / "prereg_icra.md"
    prereg.write_text("**Status:** Outcome-blind protocol draft\n")
    monkeypatch.setattr(
        barn_campaign, "BarnGazeboSpace", _FakeBarnGazeboSpace)
    common = dict(
        dataset_root=dataset,
        manifest_path=manifest,
        split_path=split,
        prereg_path=prereg,
        robot_sdf=robot,
        runtime_root=tmp_path / "runtime",
        seed=3,
        steps=1,
        n_rollouts=2,
        tasks_per_step=1,
        eval_every=1,
        eval_episodes=1,
        smoke=False,
        expected_manifest_sha256=barn_campaign._sha256(manifest),
        expected_split_sha256=barn_campaign._sha256(split),
        expected_analyzer_sha256=barn_campaign._sha256(
            barn_campaign.DEFAULT_ANALYZER),
        expected_protocol_sha256=barn_campaign._sha256(
            barn_campaign.DEFAULT_PROTOCOL),
        container_sha256="a" * 64,
        source_sha256="b" * 64,
        campaign_id="test-primary",
        attempt_id="attempt-001",
        submitted_utc="2026-08-14T00:00:00Z",
        slurm_job_id="12345_3",
        slurm_array_job_id="12345",
        slurm_array_task_id=3,
        training_sim_step_budget=14,
        eval_sim_step_interval=14,
    )
    with pytest.raises(ValueError, match="expected manifest, split"):
        barn_campaign.run_campaign(**common)
    common["expected_prereg_sha256"] = barn_campaign._sha256(prereg)
    with pytest.raises(ValueError, match="marked FROZEN"):
        barn_campaign.run_campaign(**common)


def test_frozen_prereg_status_requires_exact_line(tmp_path):
    prereg = tmp_path / "prereg.md"
    for invalid in (
        "**Status:** FROZEN, candidate\n",
        "**Status:**  FROZEN\n",
    ):
        prereg.write_text(invalid)
        with pytest.raises(ValueError, match="marked FROZEN"):
            barn_campaign._verify_frozen_prereg(
                prereg, barn_campaign._sha256(prereg))
    prereg.write_text("**Status:** FROZEN\n")
    assert barn_campaign._verify_frozen_prereg(
        prereg, barn_campaign._sha256(prereg)) == barn_campaign._sha256(prereg)


def test_evaluation_policy_mutation_is_rejected(monkeypatch, tmp_path):
    class MutatingFake(_FakeBarnGazeboSpace):
        instances = []
        eval_calls = []

        def evaluate_course(self, course, n, *, seed):
            result = super().evaluate_course(course, n, seed=seed)
            self.policy.weight += 1.0
            return result

    dataset, manifest, split, robot = _frozen_fixture(tmp_path)
    frozen = json.loads(split.read_text())
    monkeypatch.setattr(barn_campaign, "BarnGazeboSpace", MutatingFake)
    with pytest.raises(RuntimeError, match="mutated training state"):
        barn_campaign.run_campaign(
            dataset_root=dataset,
            manifest_path=manifest,
            split_path=split,
            robot_sdf=robot,
            runtime_root=tmp_path / "runtime",
            seed=3,
            arms=("ours_uN",),
            steps=1,
            n_rollouts=2,
            tasks_per_step=1,
            eval_every=1,
            eval_episodes=2,
            smoke=True,
            engineering_course_id=frozen["train_ids"][0],
        )


def test_cli_accepts_hopper_flag_spellings(tmp_path):
    parser = barn_campaign.build_parser()
    args = parser.parse_args([
        "--backend", "barn_gazebo",
        "--dataset-root", str(tmp_path / "dataset"),
        "--split", str(tmp_path / "split.json"),
        "--seed", "4",
        "--out", str(tmp_path / "seed4.json"),
    ])
    assert args.seed == 4
    assert args.output == tmp_path / "seed4.json"
    assert args.backend == "barn_gazebo"


def test_engineering_course_smoke_uses_training_partition_only(
        monkeypatch, tmp_path):
    dataset, manifest, split, robot = _frozen_fixture(tmp_path)
    frozen = json.loads(split.read_text())
    course_id = frozen["train_ids"][0]
    rows = {
        row["env_id"]: row
        for row in map(json.loads, manifest.read_text().splitlines())
    }
    # A train-only smoke must succeed even if no prospective held-out asset is
    # available to resolve, stat, hash, or load.
    for heldout_id in frozen["heldout_ids"]:
        (dataset / rows[heldout_id]["asset"]).unlink()
        (dataset / rows[heldout_id]["path_asset"]).unlink()
        (dataset / rows[heldout_id]["grid_asset"]).unlink()
    _FakeBarnGazeboSpace.reset()
    monkeypatch.setattr(
        barn_campaign, "BarnGazeboSpace", _FakeBarnGazeboSpace)

    artifact = barn_campaign.run_campaign(
        dataset_root=dataset,
        manifest_path=manifest,
        split_path=split,
        robot_sdf=robot,
        runtime_root=tmp_path / "runtime",
        seed=3,
        arms=("ours_uN",),
        steps=1,
        n_rollouts=2,
        tasks_per_step=1,
        eval_every=1,
        eval_episodes=1,
        smoke=True,
        engineering_course_id=course_id,
    )

    assert artifact["evidence_status"] == (
        "engineering_smoke_not_paper_evidence")
    assert artifact["config"]["engineering_course_id"] == course_id
    assert artifact["config"]["evaluation_partition"] == (
        "training_course_engineering_smoke")
    assert artifact["config"]["n_train_courses"] == 1
    assert artifact["config"]["n_heldout_courses"] == 1
    assert {call["course"] for call in _FakeBarnGazeboSpace.eval_calls} == {
        course_id}
    assert set(frozen["heldout_ids"]).isdisjoint(
        {call["course"] for call in _FakeBarnGazeboSpace.eval_calls})


def test_engineering_course_override_rejects_heldout_or_evidence(
        monkeypatch, tmp_path):
    dataset, manifest, split, robot = _frozen_fixture(tmp_path)
    frozen = json.loads(split.read_text())
    monkeypatch.setattr(
        barn_campaign, "BarnGazeboSpace", _FakeBarnGazeboSpace)

    with pytest.raises(ValueError, match="training partition"):
        barn_campaign.run_campaign(
            dataset_root=dataset,
            manifest_path=manifest,
            split_path=split,
            robot_sdf=robot,
            seed=3,
            arms=("ours_uN",),
            steps=1,
            n_rollouts=2,
            tasks_per_step=1,
            eval_every=1,
            eval_episodes=1,
            smoke=True,
            engineering_course_id=frozen["heldout_ids"][0],
        )
    with pytest.raises(ValueError, match="only for non-evidentiary smoke"):
        barn_campaign.run_campaign(
            dataset_root=dataset,
            manifest_path=manifest,
            split_path=split,
            robot_sdf=robot,
            seed=3,
            engineering_course_id=frozen["train_ids"][0],
            smoke=False,
        )


def test_machine_protocol_rejects_schedule_drift(tmp_path):
    _, manifest, split, _ = _frozen_fixture(tmp_path)
    protocol = _frozen_protocol(tmp_path, manifest, split)
    digest, _, contract = barn_campaign._verify_frozen_protocol(
        protocol, barn_campaign._sha256(protocol), "primary")
    assert digest == barn_campaign._sha256(protocol)
    actual = dict(contract)
    actual["n_rollouts"] = contract["n_rollouts"] * 2
    with pytest.raises(ValueError, match="differs from frozen protocol"):
        barn_campaign._enforce_protocol_contract(contract, actual)

    payload = json.loads(protocol.read_text())
    payload["status"] = "DRAFT"
    protocol.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="marked FROZEN"):
        barn_campaign._verify_frozen_protocol(
            protocol, barn_campaign._sha256(protocol), "primary")


def test_sim_step_auc_is_interpolated_at_exact_budget():
    history = [
        {"sim_steps": 0, "eval": {"mean_success": 0.0}},
        {"sim_steps": 80, "eval": {"mean_success": 0.8}},
        {"sim_steps": 120, "eval": {"mean_success": 1.0}},
    ]
    # At step 100 the interpolated success is 0.9; trapezoids contribute
    # 32 + 17 = 49, normalized by the exact budget 100.
    assert barn_campaign._normalized_auc(
        history, "sim_steps", budget=100) == pytest.approx(0.49)


def test_execution_identity_and_atomic_publication_are_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="execution identity"):
        barn_campaign._execution_identity(
            campaign_id=None,
            attempt_id=None,
            submitted_utc=None,
            slurm_job_id=None,
            slurm_array_job_id=None,
            slurm_array_task_id=None,
        )
    identity = barn_campaign._execution_identity(
        campaign_id="primary-v1",
        attempt_id="attempt-001",
        submitted_utc="2026-08-14T00:00:00Z",
        slurm_job_id="12345_1",
        slurm_array_job_id="12345",
        slurm_array_task_id=1,
    )
    assert identity["submitted_utc"].endswith("+00:00")

    output = tmp_path / "artifact.json"
    barn_campaign._atomic_write_json(output, {"finite": 1.0})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        barn_campaign._atomic_write_json(output, {"finite": 2.0})
    with pytest.raises(ValueError, match="Out of range float values"):
        barn_campaign._atomic_write_json(
            tmp_path / "nonfinite.json", {"bad": float("nan")})
