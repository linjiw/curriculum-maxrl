"""CPU tests for the frontier-curriculum Isaac Lab terms — no isaaclab/sim needed.

Run:  python3 -m pytest scripts/curriculum-maxrl/isaaclab_integration/test_frontier_terms.py -q
(or plain `python3 test_frontier_terms.py`)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from isaaclab_integration.frontier_terms import (  # noqa: E402
    FixedLevelProbe,
    FrontierTerrainTeacher,
    ScriptedTerrainLevels,
    StaticTerrainLevels,
    UniformTerrainLevels,
    apply_arm,
    SUCCESS_FNS,
)
from frontier_rl.adapters.isaaclab_curriculum import FrontierBinTeacher  # noqa: E402
from frontier_rl.adapters.isaaclab_curriculum import FrontierTerrainTeacherTerm  # noqa: E402

N_ENVS, N_LEVELS, N_TYPES = 256, 10, 4


# ---------------------------------------------------------------------------
# Stub env replicating the exact attributes the terms touch
# ---------------------------------------------------------------------------

class StubTerrain:
    def __init__(self):
        self.max_terrain_level = N_LEVELS
        self.terrain_levels = torch.randint(0, N_LEVELS, (N_ENVS,))
        self.terrain_types = torch.randint(0, N_TYPES, (N_ENVS,))
        self.terrain_origins = torch.randn(N_LEVELS, N_TYPES, 3)
        self.env_origins = self.terrain_origins[self.terrain_levels, self.terrain_types]
        self.cfg = types.SimpleNamespace(
            terrain_generator=types.SimpleNamespace(size=(8.0, 8.0)))


class StubScene:
    def __init__(self):
        self.terrain = StubTerrain()
        self._robot = types.SimpleNamespace(
            data=types.SimpleNamespace(root_pos_w=torch.zeros(N_ENVS, 3)))

    @property
    def env_origins(self):
        return self.terrain.env_origins

    def __getitem__(self, name):
        assert name == "robot"
        return self._robot


class StubEnv:
    """Latent difficulty model: success probability falls linearly with level."""

    def __init__(self, seed=0, log_dir=None):
        torch.manual_seed(seed)  # stub terrain init uses the global torch RNG
        self.scene = StubScene()
        self.termination_manager = types.SimpleNamespace(
            time_outs=torch.zeros(N_ENVS, dtype=torch.bool),
            terminated=torch.zeros(N_ENVS, dtype=torch.bool))
        self.command_manager = types.SimpleNamespace(
            get_command=lambda name: self._command)
        self._command = torch.ones(N_ENVS, 3) * 0.7  # ~1.0 m/s commanded speed
        self.max_episode_length_s = 20.0
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(N_ENVS, dtype=torch.long)
        self.cfg = types.SimpleNamespace(seed=seed, log_dir=log_dir)
        self.rng = np.random.default_rng(seed)

    def simulate_outcomes(self, env_ids):
        """Set time_outs ~ Bernoulli(p(level)): p = 0.95 - 0.1*level."""
        self.episode_length_buf[env_ids] = 100  # episodes actually ran
        levels = self.scene.terrain.terrain_levels[env_ids].numpy()
        p = np.clip(0.95 - 0.1 * levels, 0.0, 1.0)
        success = torch.from_numpy(self.rng.random(len(env_ids)) < p)
        self.termination_manager.time_outs[env_ids] = success
        # position robots: successful ones walked 6 m, failed ones 1 m
        dist = torch.where(success, 6.0, 1.0)
        origins = self.scene.env_origins[env_ids]
        self.scene._robot.data.root_pos_w[env_ids, 0] = origins[:, 0] + dist
        self.scene._robot.data.root_pos_w[env_ids, 1] = origins[:, 1]


def make_term(cls):
    return cls(cfg=types.SimpleNamespace(params={}), env=None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_static_moves_nothing():
    env = StubEnv()
    before = env.scene.terrain.terrain_levels.clone()
    term = make_term(StaticTerrainLevels)
    out = term(env, torch.arange(64))
    assert torch.equal(env.scene.terrain.terrain_levels, before)
    assert float(out) == before.float().mean()


def test_scripted_ramp_and_origins():
    env = StubEnv()
    term = make_term(ScriptedTerrainLevels)
    env.common_step_counter = 0
    out = term(env, torch.arange(N_ENVS), total_steps=1000)
    assert out["allowed_max"] == 1
    assert env.scene.terrain.terrain_levels.max() == 0
    env.common_step_counter = 500
    out = term(env, torch.arange(N_ENVS), total_steps=1000)
    assert out["allowed_max"] == N_LEVELS // 2
    assert env.scene.terrain.terrain_levels.max() < N_LEVELS // 2
    env.common_step_counter = 2000
    out = term(env, torch.arange(N_ENVS), total_steps=1000)
    assert out["allowed_max"] == N_LEVELS
    # origins must track levels
    t = env.scene.terrain
    assert torch.allclose(t.env_origins, t.terrain_origins[t.terrain_levels, t.terrain_types])


def test_uniform_covers_all_bins():
    env = StubEnv()
    term = make_term(UniformTerrainLevels)
    term(env, torch.arange(N_ENVS))
    seen = set(env.scene.terrain.terrain_levels.tolist())
    assert seen == set(range(N_LEVELS))


def test_teacher_tracks_latent_difficulty_and_writes_origins():
    env = StubEnv(seed=1)
    term = make_term(FrontierTerrainTeacher)
    ids = torch.arange(N_ENVS)
    for _ in range(60):
        env.simulate_outcomes(ids)
        out = term(env, ids, success_fn="survival", save_every_calls=0)
    # posterior should recover the p = 0.95 - 0.1*level gradient
    est = term.teacher.pass_rate_estimates()
    true = np.clip(0.95 - 0.1 * np.arange(N_LEVELS), 0.0, 1.0)
    err = np.abs(est - true)
    assert err.mean() < 0.15, f"posterior off: {est} vs {true}"
    # frontier (learnability peak) should be where p is nearest 0.5 → level 4-5
    assert out["frontier_bin"] in (3.0, 4.0, 5.0, 6.0)
    # sampling mass concentrates in the ZPD band vs a uniform sampler (P-A mechanism)
    probs = term.teacher.sampling_probs()
    zpd = [i for i in range(N_LEVELS) if 0.2 < true[i] < 0.8]
    assert probs[zpd].sum() > len(zpd) / N_LEVELS + 0.1
    # origins consistent with written levels
    t = env.scene.terrain
    assert torch.allclose(t.env_origins, t.terrain_origins[t.terrain_levels, t.terrain_types])
    # telemetry keys present and scalar-like
    for key in ("mean_bin", "frontier_bin", "dead_frac", "mastered_frac",
                "effective_bins", "seen_frac", "zpd_mass", "zpd_bins"):
        assert key in out
    # the logged zpd_mass IS the P-A gate quantity: > uniform share of ZPD bins
    assert out["zpd_mass"] > out["zpd_bins"] / N_LEVELS


def test_deterministic_optimism_maximizes_utility_over_confidence_interval():
    teacher = FrontierBinTeacher(
        n_bins=4, utility="learnability", optimism_k=1.0, thompson=False
    )
    # An unseen Beta(1,1) posterior has mean 0.5. A naive mean+std transform
    # moves p to 0.789 and lowers p(1-p); utility-space optimism stays at 0.5.
    assert np.allclose(teacher._utility(), np.full(4, 0.25))

    teacher.succ[0], teacher.fail[0] = 99.0, 1.0
    teacher._dirty = True
    mean_utility = teacher.pass_rate_estimates()[0] * (1 - teacher.pass_rate_estimates()[0])
    assert teacher._utility()[0] >= mean_utility


def test_probability_ceiling_is_a_tripwire_not_distribution_shaping():
    teacher = FrontierBinTeacher(n_bins=4, max_prob=0.2)
    try:
        teacher.sample_bins(1)
        raise AssertionError("tripwire should fire because uniform probability 0.25 exceeds 0.2")
    except RuntimeError as exc:
        assert "tripwire fired" in str(exc)
    with np.testing.assert_raises(ValueError):
        teacher.max_prob = 1.01


def test_bin_teacher_state_restores_exact_sampling_stream():
    teacher = FrontierBinTeacher(n_bins=5, thompson=True, seed=12)
    teacher.observe_resets(np.array([0, 1, 2, 3]), np.array([False, True, False, True]))
    teacher.sampling_probs()
    state = teacher.state_dict()
    expected = teacher.sample_bins(30)

    restored = FrontierBinTeacher(n_bins=5, thompson=True, seed=999)
    restored.load_state_dict(state)
    assert np.array_equal(restored.sample_bins(30), expected)
    with np.testing.assert_raises_regex(ValueError, "configuration mismatch"):
        FrontierBinTeacher(n_bins=5, thompson=False).load_state_dict(state)


def test_teacher_ignores_construction_reset():
    """The wrapper's construction-time full reset (no episodes run yet) must not
    poison the posterior with fake failures."""
    env = StubEnv(seed=4)
    term = make_term(FrontierTerrainTeacher)
    ids = torch.arange(N_ENVS)
    # construction reset: episode_length_buf all zero, time_outs all False
    term(env, ids, save_every_calls=0)
    assert term.teacher.succ.sum() == 0 and term.teacher.fail.sum() == 0
    # a real reset afterwards is observed
    env.simulate_outcomes(ids)
    term(env, ids, save_every_calls=0)
    assert term.teacher.succ.sum() + term.teacher.fail.sum() > 0


def test_teacher_and_probe_filter_mixed_zero_length_resets():
    """A mixed reset batch must observe only envs whose episodes actually ran."""
    ids = torch.arange(N_ENVS)
    completed = ids[: N_ENVS // 2]

    env = StubEnv(seed=14)
    env.simulate_outcomes(completed)
    term = make_term(FrontierTerrainTeacher)
    term(env, ids, success_fn="survival", save_every_calls=0)
    evidence = term.teacher.succ.sum() + term.teacher.fail.sum()
    assert evidence == len(completed)

    env2 = StubEnv(seed=14)
    env2.simulate_outcomes(completed)
    probe = make_term(FixedLevelProbe)
    probe(env2, ids, success_fn="survival")
    assert probe.succ.sum() + probe.fail.sum() == len(completed)


def test_reusable_adapter_ignores_construction_reset_and_loads_lazily():
    env = StubEnv(seed=9)
    ids = torch.arange(N_ENVS)
    term = FrontierTerrainTeacherTerm(seed=9)
    term(env, ids)
    assert term.teacher.succ.sum() == 0 and term.teacher.fail.sum() == 0
    env.simulate_outcomes(ids)
    term(env, ids)
    state = term.state_dict()
    expected = term.teacher.sample_bins(N_ENVS)

    env2 = StubEnv(seed=10)
    restored = FrontierTerrainTeacherTerm(seed=999)
    restored.load_state_dict(state)
    restored(env2, ids)
    assert np.array_equal(env2.scene.terrain.terrain_levels.numpy(), expected)


def test_fixed_level_probe():
    env = StubEnv(seed=5)
    probe = make_term(FixedLevelProbe)
    ids = torch.arange(N_ENVS)
    # construction reset: pins levels, tallies nothing
    out = probe(env, ids)
    assert out["episodes"] == 0
    assert torch.equal(env.scene.terrain.terrain_levels, ids % N_LEVELS)
    t = env.scene.terrain
    assert torch.allclose(t.env_origins, t.terrain_origins[t.terrain_levels, t.terrain_types])
    # run episodes on the pinned grid; probe tallies by level
    for _ in range(20):
        env.simulate_outcomes(ids)
        probe(env, ids, success_fn="survival")
    res = probe.results()
    # near-balanced grid: env i % n_levels → levels 0-5 get 26 envs, 6-9 get 25
    expected = [20.0 * ((N_ENVS // N_LEVELS) + (1 if lvl < N_ENVS % N_LEVELS else 0))
                for lvl in range(N_LEVELS)]
    assert res["episodes_per_level"] == expected
    # measured per-level pass should recover p = 0.95 - 0.1*level
    true = np.clip(0.95 - 0.1 * np.arange(N_LEVELS), 0.0, 1.0)
    meas = np.array(res["per_level_pass"], dtype=float)
    assert np.abs(meas - true).mean() < 0.1, f"{meas} vs {true}"


def test_fixed_level_probe_reports_macro_average():
    probe = make_term(FixedLevelProbe)
    probe.succ = np.zeros(N_LEVELS)
    probe.fail = np.zeros(N_LEVELS)
    probe.succ[:2] = [1.0, 9.0]
    probe.fail[:2] = [0.0, 1.0]
    result = probe.results()
    assert abs(result["mean_pass"] - 0.95) < 1e-12
    assert abs(result["micro_pass"] - 10.0 / 11.0) < 1e-12


def test_teacher_success_predicates():
    env = StubEnv(seed=2)
    ids = torch.arange(N_ENVS)
    env.simulate_outcomes(ids)
    surv = SUCCESS_FNS["survival"](env, ids)
    assert torch.equal(surv, env.termination_manager.time_outs[ids])
    # distance: commanded speed ‖(0.7,0.7)‖≈0.99 m/s → required = 0.99*20*f.
    # At f=0.25 required ≈ 4.95 m: success stubs walked 6 m (pass), failures 1 m (fail).
    from isaaclab_integration.frontier_terms import _success_distance
    dist = _success_distance(env, ids, fraction=0.25)
    assert torch.equal(dist, surv)
    # at f=0.5 (required ≈ 9.9 m) even the 6 m walkers fail
    assert not _success_distance(env, ids, fraction=0.5).any()
    # tile: 6 m > 8/2 = 4 m → same set
    tile = SUCCESS_FNS["tile"](env, ids)
    assert torch.equal(tile, surv)


def test_teacher_state_checkpoint_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        env = StubEnv(seed=3, log_dir=tmp)
        term = make_term(FrontierTerrainTeacher)
        ids = torch.arange(N_ENVS)
        for _ in range(5):
            env.simulate_outcomes(ids)
            term(env, ids, save_every_calls=5)
        path = os.path.join(tmp, "curriculum_teacher", "teacher_state.json")
        assert os.path.exists(path)
        with open(path) as f:
            state = json.load(f)
        assert len(state["teacher"]["succ"]) == N_LEVELS
        assert "rng_state" in state["teacher"]
        # resume into a fresh term
        env2 = StubEnv(seed=3)
        term2 = make_term(FrontierTerrainTeacher)
        term2(env2, ids, load_state=path, save_every_calls=0)
        # The construction reset is ignored, so the loaded posterior is unchanged.
        assert term2.teacher.succ.sum() > 0
        assert term2._calls == state["calls"] + 1


def test_teacher_determinism_same_seed():
    outs = []
    for _ in range(2):
        env = StubEnv(seed=7)
        term = make_term(FrontierTerrainTeacher)
        ids = torch.arange(N_ENVS)
        for _ in range(10):
            env.simulate_outcomes(ids)
            term(env, ids, save_every_calls=0)
        outs.append(env.scene.terrain.terrain_levels.clone())
    assert torch.equal(outs[0], outs[1])


def test_apply_arm():
    class TermCfg:
        def __init__(self):
            self.func = "stock_terrain_levels_vel"
            self.params = {}

    class Cfg:
        def __init__(self):
            self.curriculum = types.SimpleNamespace(terrain_levels=TermCfg())

    cfg = Cfg()
    apply_arm(cfg, "greedy")
    assert cfg.curriculum.terrain_levels.func == "stock_terrain_levels_vel"
    apply_arm(cfg, "teacher", success_fn="tile", teacher_params={"floor": 0.2})
    assert cfg.curriculum.terrain_levels.func is FrontierTerrainTeacher
    assert cfg.curriculum.terrain_levels.params == {"success_fn": "tile", "floor": 0.2}
    apply_arm(cfg, "scripted", scripted_total_steps=123)
    assert cfg.curriculum.terrain_levels.params == {"total_steps": 123}
    apply_arm(cfg, "control")
    assert cfg.curriculum.terrain_levels.func is StaticTerrainLevels
    try:
        apply_arm(cfg, "bogus")
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_term_parameters_fail_loudly():
    env = StubEnv()
    ids = torch.arange(N_ENVS)
    with np.testing.assert_raises_regex(ValueError, "total_steps"):
        make_term(ScriptedTerrainLevels)(env, ids, total_steps=0)
    with np.testing.assert_raises_regex(ValueError, "success_fn"):
        make_term(FrontierTerrainTeacher)(env, ids, success_fn="unknown")
    del env.episode_length_buf
    with np.testing.assert_raises_regex(RuntimeError, "episode_length_buf"):
        make_term(FixedLevelProbe)(env, ids)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
