"""GPU curriculum x MaxRL experiment on multi-size mazes.

Pipeline per run:
  1. SFT warmstart on level-0/1 BFS solutions only (so deeper levels start
     near p=0 and the curriculum question is real).
  2. RL loop: teacher picks levels -> sample fresh mazes (infinite-data
     regime, as in the paper's maze experiment) -> group rollouts -> binary
     verifier -> estimator advantages -> policy-gradient step.
  3. Periodic eval on a fixed held-out set per level (pass@1 greedy-free,
     sampled) -> results JSONL.

Usage:
  python3 train.py --teacher uniform --estimator maxrl --steps 300 --seed 0
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maze_env import (LEVELS, MOVE_BUDGET, PAD, EOS,
                      sample_task, sft_example, verify, simulate_prefix,
                      encode_prompt)
from model import TinyTransformer
from estimators import (coefficient_activity, legacy_frontier_activity,
                        weights_reinforce, weights_rloo, weights_grpo,
                        weights_maxrl)

ESTIMATORS = {
    "reinforce": weights_reinforce,
    "rloo": weights_rloo,
    "grpo": weights_grpo,
    "maxrl": weights_maxrl,
}

DEVICE = "cuda"
PROTOCOLS = ("legacy_v1", "maze_score_v2")
MAZE_SCORE_V2_STEPS = 250
MAZE_SCORE_V2_EVAL_EVERY = 25
MAZE_SCORE_V2_EVAL_TASKS_PER_LEVEL = 32
MAZE_SCORE_V2_EVAL_SAMPLES = 8
MAZE_SCORE_V2_EVAL_TASK_SEED_BASE = 202_608_130
MAZE_SCORE_V2_EVAL_SAMPLE_SEED_BASE = 302_608_130


def resolve_sft_checkpoint(protocol: str, sft_ckpt: str, seed: int,
                           script_dir: str | os.PathLike | None = None) -> Path:
    """Resolve a warmstart path without changing ``legacy_v1`` semantics.

    MAZE-SCORE v2 requires a deliberate absolute path and uses that path
    exactly (without adding the seed or script directory).  Requiring the
    parent up front turns a common scratch/staging typo into a launch error.
    """
    if protocol == "maze_score_v2":
        path = Path(sft_ckpt)
        if not path.is_absolute():
            raise ValueError("maze_score_v2 requires an absolute --sft-ckpt")
        if not path.parent.is_dir():
            raise FileNotFoundError(
                f"parent of --sft-ckpt does not exist: {path.parent}")
        return path
    if protocol != "legacy_v1":
        raise ValueError(f"unknown protocol: {protocol}")
    base = Path(script_dir or os.path.dirname(os.path.abspath(__file__)))
    return base / f"seed{seed}_{sft_ckpt}"


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_stream_seed(base_seed: int, stream: str) -> int:
    """Stable 64-bit seed derivation (independent of Python hash randomization)."""
    payload = f"maze_score_v2:{base_seed}:{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def seed_global_rngs(seed: int) -> None:
    """Reset the Python, NumPy, and Torch global streams for a named phase."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_level_task_rngs(rl_seed: int) -> dict[int, random.Random]:
    """Create one stable task stream per level for paired-arm comparisons."""
    return {
        level: random.Random(derived_stream_seed(
            rl_seed, f"rl_tasks_level_{level}"))
        for level in LEVELS
    }


def maze_score_v2_eval_schedule(steps: int, eval_every: int) -> tuple[int, ...]:
    """Return the frozen MAZE-SCORE v2 RL-evaluation schedule."""
    if steps != MAZE_SCORE_V2_STEPS or eval_every != MAZE_SCORE_V2_EVAL_EVERY:
        raise ValueError(
            "maze_score_v2 fixes --steps=250 and --eval-every=25")
    return tuple(range(eval_every, steps + 1, eval_every))


def teacher_distribution_snapshot(teacher: "Teacher") -> np.ndarray:
    """Inspect a stochastic teacher without advancing its sampling stream."""
    state = copy.deepcopy(teacher.rng.bit_generator.state)
    try:
        return teacher.distribution()
    finally:
        teacher.rng.bit_generator.state = state


def score_metadata(teacher: str, n_rollouts: int) -> tuple[str, int | None]:
    """Return the logged score family and its deployed effective exponent."""
    if teacher in {"frontier_un", "coefficient_activity"}:
        return "coefficient_activity", n_rollouts
    if teacher == "frontier":
        return "legacy_frontier_activity", n_rollouts + 1
    if teacher == "learnability":
        return "coefficient_activity", 2
    if teacher == "frontier_un_tilt":
        return "coefficient_activity_tilt", n_rollouts
    if teacher == "frontier_alp":
        return "legacy_frontier_activity_plus_alp", n_rollouts + 1
    return teacher, None


# ------------------------------------------------------------------ teachers
class Teacher:
    """Level-based teacher: curriculum over the 7 maze sizes."""

    def __init__(self, n_rollouts: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.n_rollouts = n_rollouts
        self.alpha = np.ones(len(LEVELS))
        self.beta = np.ones(len(LEVELS))

    def observe(self, level: int, rewards: np.ndarray, decay: float = 0.7):
        k, n = rewards.sum(), len(rewards)
        self.alpha[level] = 1.0 + (self.alpha[level] - 1.0) * decay + k
        self.beta[level] = 1.0 + (self.beta[level] - 1.0) * decay + (n - k)

    def p_hat(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def distribution(self) -> np.ndarray:
        raise NotImplementedError

    def sample_levels(self, m: int) -> np.ndarray:
        return self.rng.choice(len(LEVELS), size=m, p=self.distribution())


class UniformTeacher(Teacher):
    def distribution(self) -> np.ndarray:
        return np.full(len(LEVELS), 1.0 / len(LEVELS))


class FrontierTeacher(Teacher):
    """Historical shifted frontier score with Thompson p and uniform floor."""

    def __init__(self, n_rollouts: int, seed: int, floor: float = 0.15):
        super().__init__(n_rollouts, seed)
        self.floor = floor

    def distribution(self) -> np.ndarray:
        p = self.rng.beta(self.alpha, self.beta)
        u = legacy_frontier_activity(p, self.n_rollouts)
        if u.sum() <= 1e-12:
            u[:] = 1.0
        probs = u / u.sum()
        unif = np.full(len(LEVELS), 1.0 / len(LEVELS))
        return (1 - self.floor) * probs + self.floor * unif


class LearnabilityTeacher(Teacher):
    """SFL-style u(p) = p(1-p) — exact coefficient activity at N=2."""

    def __init__(self, n_rollouts: int, seed: int, floor: float = 0.15):
        super().__init__(n_rollouts, seed)
        self.floor = floor

    def distribution(self) -> np.ndarray:
        p = self.rng.beta(self.alpha, self.beta)
        u = p * (1.0 - p)
        if u.sum() <= 1e-12:
            u[:] = 1.0
        probs = u / u.sum()
        unif = np.full(len(LEVELS), 1.0 / len(LEVELS))
        return (1 - self.floor) * probs + self.floor * unif


class FrontierALPTeacher(FrontierTeacher):
    """Frontier utility + ALP-GMM-style anti-forgetting term.

    utility = u_N(p) + alp_coef * |Δ ema_pass|.  The |ΔLP| term re-injects
    levels whose competence is *changing* — including regressions on mastered
    levels, which pure u_N(p) would retire (its u -> 0 as p -> 1).

    power (VALIDATION.md V6): sample ∝ utility^power; sharper-than-
    proportional concentration compounds on ordered level structures."""

    def __init__(self, n_rollouts: int, seed: int, floor: float = 0.1,
                 alp_coef: float = 2.0, power: float = 1.0):
        super().__init__(n_rollouts, seed, floor)
        self.alp_coef = alp_coef
        self.power = power
        self.ema = np.zeros(len(LEVELS))
        self.alp = np.zeros(len(LEVELS))
        self.seen = np.zeros(len(LEVELS), dtype=bool)

    def observe(self, level: int, rewards: np.ndarray, decay: float = 0.7):
        super().observe(level, rewards, decay)
        m = rewards.mean()
        prev = self.ema[level] if self.seen[level] else m
        self.ema[level] = 0.7 * prev + 0.3 * m
        self.seen[level] = True
        self.alp[level] = 0.7 * self.alp[level] + 0.3 * abs(self.ema[level] - prev)

    def distribution(self) -> np.ndarray:
        p = self.rng.beta(self.alpha, self.beta)
        u = (legacy_frontier_activity(p, self.n_rollouts)
             + self.alp_coef * self.alp)
        u = np.maximum(u, 0.0) ** self.power
        if u.sum() <= 1e-12:
            u[:] = 1.0
        probs = u / u.sum()
        unif = np.full(len(LEVELS), 1.0 / len(LEVELS))
        return (1 - self.floor) * probs + self.floor * unif


class FrontierUNTeacher(FrontierTeacher):
    """The paper's exact derived utility u_N(p) = (1-(1-p)^N) - p.

    The legacy FrontierTeacher above uses (1-(1-p)^N)(1-p) — a
    pre-derivation heuristic (opus5 M4). Chains show the two within
    noise (v7 battery: 0.728 vs 0.733); this arm tests that on the maze.
    """

    def distribution(self) -> np.ndarray:
        p = self.rng.beta(self.alpha, self.beta)
        u = coefficient_activity(p, self.n_rollouts)
        u = np.maximum(u, 0.0)
        if u.sum() <= 1e-12:
            u[:] = 1.0
        probs = u / u.sum()
        unif = np.full(len(LEVELS), 1.0 / len(LEVELS))
        return (1 - self.floor) * probs + self.floor * unif


class FrontierUNTiltTeacher(FrontierTeacher):
    """Horizon-tilted utility (1-p) * u_N(p) (BRIDGE_ANALYSIS part E, alpha=1).

    The bridge experiments found the tilt a deployable improvement over
    plain u_N on both CPU pools through posterior noise; this arm is its
    first GPU test.
    """

    def distribution(self) -> np.ndarray:
        p = self.rng.beta(self.alpha, self.beta)
        u = (1.0 - p) * coefficient_activity(p, self.n_rollouts)
        u = np.maximum(u, 0.0)
        if u.sum() <= 1e-12:
            u[:] = 1.0
        probs = u / u.sum()
        unif = np.full(len(LEVELS), 1.0 / len(LEVELS))
        return (1 - self.floor) * probs + self.floor * unif


TEACHERS = {
    "uniform": UniformTeacher,
    "frontier": FrontierTeacher,
    "learnability": LearnabilityTeacher,
    "frontier_alp": FrontierALPTeacher,
    "frontier_un": FrontierUNTeacher,
    "coefficient_activity": FrontierUNTeacher,
    "frontier_un_tilt": FrontierUNTiltTeacher,
}


# ------------------------------------------------------------------ batching
def pad_batch(seqs: list[list[int]], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    lens = torch.tensor([len(s) for s in seqs], device=device)
    out = torch.full((len(seqs), int(lens.max())), PAD, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, device=device)
    return out, lens


def response_logprobs(model, prompts, prompt_lens, resps):
    """Sum log pi(response tokens) per sample. resps: (B, Lr) PAD after EOS."""
    B, Lr = resps.shape
    full = torch.full((B, prompts.shape[1] + Lr), PAD, dtype=torch.long,
                      device=prompts.device)
    full[:, :prompts.shape[1]] = prompts
    resp_mask = resps != PAD
    for b in range(B):
        n = int(resp_mask[b].sum())
        full[b, prompt_lens[b]:prompt_lens[b] + n] = resps[b, :n]
    logits = model(full[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    tgt = full[:, 1:]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
    # mask: positions belonging to the response
    pos = torch.arange(full.shape[1] - 1, device=prompts.device)[None]
    n_resp = resp_mask.sum(1)
    mask = (pos >= (prompt_lens - 1)[:, None]) & (pos < (prompt_lens - 1 + n_resp)[:, None])
    return (tok_lp * mask).sum(1), mask.sum(1)


# ------------------------------------------------------------------ SFT
def run_sft(model, opt, rng, steps=600, batch=64, decay=0.5,
            np_rng: np.random.Generator | None = None):
    """SFT on a geometric mixture over levels (weight decay^level): shallow
    levels dominate, deep levels are seen rarely — so post-SFT pass rates
    decay smoothly with depth instead of cliffing to exactly 0 (mirrors the
    paper's 'brief SFT to ensure non-zero initial pass rate')."""
    w = np.array([decay ** l for l in LEVELS])
    w = w / w.sum()
    model.train()
    for step in range(steps):
        if np_rng is None:
            # Historical global stream used by legacy_v1.
            lvls = np.random.choice(LEVELS, size=batch, p=w)
        else:
            lvls = np_rng.choice(LEVELS, size=batch, p=w)
        pairs = [sft_example(int(l), rng) for l in lvls]
        seqs = [p + r for p, r in pairs]
        plens = [len(p) for p, _ in pairs]
        ids, lens = pad_batch(seqs, DEVICE)
        logits = model(ids[:, :-1])
        tgt = ids[:, 1:]
        lp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        pos = torch.arange(ids.shape[1] - 1, device=DEVICE)[None]
        plens_t = torch.tensor(plens, device=DEVICE)
        mask = (pos >= (plens_t - 1)[:, None]) & (pos < (lens - 1)[:, None])
        loss = -(lp * mask).sum() / mask.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"  sft step {step} loss {loss.item():.3f}", flush=True)


# ------------------------------------------------------------------ eval
def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Chen et al. 2021 unbiased pass@k estimator: 1 - C(n-c,k)/C(n,k)."""
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


@torch.no_grad()
def evaluate(model, eval_tasks, n_samples=8, batch_cap=256, pass_ks=(1, 8),
             generator: torch.Generator | None = None):
    """Per-level sampled pass rate + unbiased pass@k on fixed held-out mazes.

    Returns {level: mean_pass} plus {"passk": {level: {k: pass@k}}} computed
    per maze from its n_samples rollouts (Chen et al. 2021 estimator).
    """
    model.eval()
    out = {}
    passk = {}
    for level, tasks in eval_tasks.items():
        per_task_c = {id(t): 0 for t in tasks}
        reps = [(t, s) for t in tasks for s in range(n_samples)]
        for i in range(0, len(reps), batch_cap):
            chunk = [t for t, _ in reps[i:i + batch_cap]]
            prompts, plens = pad_batch([t.prompt for t in chunk], DEVICE)
            resp = model.generate(prompts, plens, MOVE_BUDGET[level] + 1, EOS,
                                  generator=generator)
            for j, t in enumerate(chunk):
                toks = [int(x) for x in resp[j] if int(x) != PAD]
                per_task_c[id(t)] += verify(t.grid, t.goal, toks)
        cs = np.array(list(per_task_c.values()))
        out[level] = float(cs.sum()) / (len(tasks) * n_samples)
        passk[level] = {
            k: float(np.mean([pass_at_k_unbiased(n_samples, int(c), k) for c in cs]))
            for k in pass_ks if k <= n_samples
        }
    model.train()
    out["passk"] = passk
    return out


# ------------------------------------------------------------------ RL loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=PROTOCOLS, default="legacy_v1")
    ap.add_argument("--teacher", choices=list(TEACHERS), default="uniform")
    ap.add_argument("--estimator", choices=list(ESTIMATORS), default="maxrl")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sft-seed", type=int, default=None,
                    help="v2 SFT/model-init seed (default: --seed)")
    ap.add_argument("--rl-seed", type=int, default=None,
                    help="v2 task/rollout seed (default: --seed)")
    ap.add_argument("--eval-seed", type=int, default=None,
                    help="seed for the frozen held-out maze set")
    ap.add_argument("--eval-sample-seed", type=int, default=None,
                    help="v2-only dedicated Torch generator for eval samples")
    ap.add_argument("--tasks-per-step", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--sft-steps", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-tasks-per-level", type=int, default=None)
    ap.add_argument("--eval-samples", type=int, default=8)
    ap.add_argument("--max-seconds", type=int, default=None,
                    help="stop after this much RL wall-clock (matched-compute comparisons)")
    ap.add_argument("--hindsight", action="store_true",
                    help="relabel dead (K=0) groups to the deepest cell reached")
    ap.add_argument("--hindsight-scale", type=float, default=1.0)
    ap.add_argument("--hindsight-dense", action="store_true",
                    help="relabel EVERY failed rollout (depth >= --hindsight-min-depth) "
                         "to its reached cell, not just the group's best")
    ap.add_argument("--hindsight-min-depth", type=int, default=6)
    ap.add_argument("--hindsight-cap", type=int, default=16,
                    help="max relabeled trajectories per step (compute bound)")
    ap.add_argument("--hindsight-to-teacher", action="store_true",
                    help="relabeled successes update the teacher posterior at the "
                         "matching distance level (curriculum rides hindsight gains)")
    ap.add_argument("--save-ckpt", type=str, default=None,
                    help="save the final model state_dict to this path")
    ap.add_argument("--teacher-power", type=float, default=1.0,
                    help="sample levels ∝ utility^power (V6: 4 for ordered levels)")
    ap.add_argument("--d-model", type=int, default=128,
                    help="model width (capacity probe: per-step legality ceiling)")
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--sft-ckpt", type=str, default="sft_warmstart.pt")
    ap.add_argument("--prepare-sft-only", action="store_true",
                    help="create/load the resolved warmstart, print its SHA256, and exit")
    ap.add_argument("--telemetry-out", type=str, default=None,
                    help="optional per-update diagnostic JSONL")
    ap.add_argument("--campaign", type=str, default=None)
    ap.add_argument("--source-manifest", type=str, default=None)
    ap.add_argument("--arm", type=str, default=None)
    args = ap.parse_args()

    is_v2 = args.protocol == "maze_score_v2"
    if is_v2 and args.sft_seed not in (None, args.seed):
        ap.error("maze_score_v2 requires --sft-seed equal to --seed")
    if is_v2 and not args.prepare_sft_only:
        try:
            eval_schedule = maze_score_v2_eval_schedule(
                args.steps, args.eval_every)
        except ValueError as exc:
            ap.error(str(exc))
        if args.max_seconds is not None:
            ap.error("maze_score_v2 uses fixed updates and forbids --max-seconds")
        if args.eval_seed is None or args.eval_sample_seed is None:
            ap.error("maze_score_v2 requires explicit --eval-seed and "
                     "--eval-sample-seed")
        expected_eval_seed = MAZE_SCORE_V2_EVAL_TASK_SEED_BASE + args.seed
        expected_eval_sample_seed = (
            MAZE_SCORE_V2_EVAL_SAMPLE_SEED_BASE + args.seed)
        if args.eval_seed != expected_eval_seed:
            ap.error(
                f"maze_score_v2 requires --eval-seed={expected_eval_seed}")
        if args.eval_sample_seed != expected_eval_sample_seed:
            ap.error("maze_score_v2 requires --eval-sample-seed="
                     f"{expected_eval_sample_seed}")
        if args.rl_seed not in (None, args.seed):
            ap.error("maze_score_v2 requires --rl-seed equal to --seed")
        if args.eval_tasks_per_level != MAZE_SCORE_V2_EVAL_TASKS_PER_LEVEL:
            ap.error("maze_score_v2 requires --eval-tasks-per-level=32")
        if args.eval_samples != MAZE_SCORE_V2_EVAL_SAMPLES:
            ap.error("maze_score_v2 requires --eval-samples=8")
        frozen_values = {
            "--estimator": (args.estimator, "maxrl"),
            "--rollouts": (args.rollouts, 32),
            "--tasks-per-step": (args.tasks_per_step, 8),
            "--sft-steps": (args.sft_steps, 600),
            "--d-model": (args.d_model, 128),
            "--n-layers": (args.n_layers, 6),
            "--teacher-power": (args.teacher_power, 1.0),
        }
        for option, (actual, expected) in frozen_values.items():
            if actual != expected:
                ap.error(
                    f"maze_score_v2 requires {option}={expected}, got {actual}")
        if args.lr != 1e-4:
            ap.error(f"maze_score_v2 requires --lr=0.0001, got {args.lr}")
        if args.hindsight or args.hindsight_dense or args.hindsight_to_teacher:
            ap.error("maze_score_v2 forbids all hindsight options")
        for option, value in (("--campaign", args.campaign),
                              ("--source-manifest", args.source_manifest),
                              ("--arm", args.arm)):
            if value is None or not value.strip():
                ap.error(f"maze_score_v2 requires nonempty {option}")
        expected_teachers = {
            "un": {"frontier_un", "coefficient_activity"},
            "learn": {"learnability"},
            "unif": {"uniform"},
        }
        if args.arm not in expected_teachers:
            ap.error("maze_score_v2 --arm must be one of: un, learn, unif")
        if args.teacher not in expected_teachers[args.arm]:
            ap.error(
                f"maze_score_v2 arm {args.arm!r} is incompatible with "
                f"teacher {args.teacher!r}")
    elif is_v2:
        # SFT preparation intentionally has no arm/evaluation output contract.
        eval_schedule = ()
    else:
        eval_schedule = ()
        if args.eval_seed is None:
            args.eval_seed = 12345
        if args.eval_tasks_per_level is None:
            args.eval_tasks_per_level = 16

    sft_seed = args.seed if args.sft_seed is None else args.sft_seed
    rl_seed = args.seed if args.rl_seed is None else args.rl_seed

    # Validate the checkpoint target before allocating a model/GPU.
    try:
        ckpt = resolve_sft_checkpoint(args.protocol, args.sft_ckpt, args.seed)
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))

    # legacy_v1 deliberately keeps the original shared global streams.  V2
    # starts in a named SFT phase, then resets every RL stream after checkpoint
    # creation/loading so both paths enter RL in exactly the same RNG state.
    phase_seed = sft_seed if is_v2 else args.seed
    seed_global_rngs(phase_seed)
    rng = random.Random(phase_seed)
    sft_np_rng = np.random.default_rng(sft_seed) if is_v2 else None

    model = TinyTransformer(d_model=args.d_model, n_layers=args.n_layers).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M", flush=True)

    # ---- SFT warmstart (shared across runs with the same seed) ----
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        print(f"loaded SFT checkpoint {ckpt}", flush=True)
    else:
        sft_opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        run_sft(model, sft_opt, rng, steps=args.sft_steps, np_rng=sft_np_rng)
        torch.save(model.state_dict(), ckpt)
        print(f"saved SFT checkpoint {ckpt}", flush=True)
    sft_sha256 = sha256_file(ckpt)
    if is_v2 or args.prepare_sft_only:
        print(f"SFT_SHA256 {sft_sha256} {ckpt}", flush=True)
    if args.prepare_sft_only:
        return

    if is_v2:
        seed_global_rngs(rl_seed)
        rl_task_rngs = make_level_task_rngs(rl_seed)
        rollout_generator = torch.Generator(device=DEVICE)
    else:
        rl_task_rngs = None
        rollout_generator = None

    # Construct the RL optimizer after the v2 phase reset.  AdamW has no state
    # until its first step, making create/load warmstart paths identical.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Fixed held-out set, independent of both SFT and RL task sampling.
    eval_rng = random.Random(args.eval_seed)
    eval_tasks = {
        l: [sample_task(l, eval_rng) for _ in range(args.eval_tasks_per_level)]
        for l in LEVELS
    }

    teacher_seed = (rl_seed if is_v2 else args.seed) + 77
    teacher_kwargs = {"n_rollouts": args.rollouts, "seed": teacher_seed}
    if args.teacher == "frontier_alp" and args.teacher_power != 1.0:
        teacher_kwargs["power"] = args.teacher_power
    teacher = TEACHERS[args.teacher](**teacher_kwargs)
    est = ESTIMATORS[args.estimator]

    out_path = args.out or f"log_{args.teacher}_{args.estimator}_s{args.seed}.jsonl"
    log_f = open(out_path, "w")
    telemetry_f = open(args.telemetry_out, "w") if args.telemetry_out else None

    if is_v2:
        score_family, effective_exponent = score_metadata(
            args.teacher, args.rollouts)
        config = {
            "record_type": "config",
            "protocol": args.protocol,
            "campaign": args.campaign,
            "source_manifest": args.source_manifest,
            "arm": args.arm,
            "seed": args.seed,
            "teacher": args.teacher,
            "estimator": args.estimator,
            "score_family": score_family,
            "rollouts": args.rollouts,
            "effective_exponent": effective_exponent,
            "seeds": {
                "base": args.seed,
                "sft": sft_seed,
                "rl": rl_seed,
                "teacher": teacher_seed,
                "eval_tasks": args.eval_seed,
                "eval_samples": args.eval_sample_seed,
            },
            "steps": args.steps,
            "tasks_per_step": args.tasks_per_step,
            "lr": args.lr,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "eval_every": args.eval_every,
            "eval_tasks_per_level": args.eval_tasks_per_level,
            "eval_samples": args.eval_samples,
            "planned_rl_eval_count": len(eval_schedule),
            "sft_steps": args.sft_steps,
            "teacher_power": args.teacher_power,
            "hindsight": args.hindsight,
            "hindsight_dense": args.hindsight_dense,
            "hindsight_to_teacher": args.hindsight_to_teacher,
            "hindsight_scale": args.hindsight_scale,
            "hindsight_min_depth": args.hindsight_min_depth,
            "hindsight_cap": args.hindsight_cap,
            "sft_checkpoint": str(ckpt),
            "sft_checkpoint_sha256": sft_sha256,
        }
        log_f.write(json.dumps(config) + "\n")
        log_f.flush()

    if is_v2:
        initial_eval_generator = torch.Generator(device=DEVICE)
        initial_eval_generator.manual_seed(args.eval_sample_seed)
    else:
        initial_eval_generator = None
    ev = evaluate(model, eval_tasks, n_samples=args.eval_samples,
                  generator=initial_eval_generator)
    passk0 = ev.pop("passk")
    if is_v2:
        print("post-SFT evaluation complete (endpoints sealed in JSONL)",
              flush=True)
        initial_rec = {
            "record_type": "evaluation",
            "protocol": args.protocol,
            "phase": "post_sft",
            "completed_updates": 0,
            "eval": ev,
            "passk": passk0,
        }
    else:
        print(f"post-SFT eval: {ev}", flush=True)
        initial_rec = {"step": -1, "eval": ev, "passk": passk0}
    log_f.write(json.dumps(initial_rec) + "\n")
    log_f.flush()

    t0 = time.time()
    max_new = MOVE_BUDGET[LEVELS[-1]] + 1
    step = -1
    optimizer_steps = 0
    while True:
        step += 1
        if args.max_seconds is not None:
            if time.time() - t0 >= args.max_seconds:
                break
        elif step >= args.steps:
            break
        completed_updates = step + 1
        levels = [int(x) for x in teacher.sample_levels(args.tasks_per_step)]
        if is_v2:
            tasks = [sample_task(lv, rl_task_rngs[lv]) for lv in levels]
            # Restart every update so arm-dependent early EOS cannot shift a
            # later update's rollout-sampling stream.
            rollout_generator.manual_seed(rl_seed + completed_updates)
        else:
            tasks = [sample_task(lv, rng) for lv in levels]
        # one batched generation for all groups: (tasks*rollouts) sequences
        flat_prompts = [t.prompt for t in tasks for _ in range(args.rollouts)]
        prompts, plens = pad_batch(flat_prompts, DEVICE)
        resp = model.generate(prompts, plens, max_new, EOS,
                              generator=rollout_generator)

        step_stats = {"dead_groups": 0, "mean_reward": [], "relabeled": 0,
                      "group_k": [], "coefficient_mass": []}
        keep_rows, keep_w = [], []
        hs_prompts, hs_resps, hs_depths = [], [], []  # hindsight-relabeled
        for g, (lv, task) in enumerate(zip(levels, tasks)):
            rows = range(g * args.rollouts, (g + 1) * args.rollouts)
            rewards = np.array([
                float(verify(task.grid, task.goal,
                             [int(x) for x in resp[j] if int(x) != PAD]))
                for j in rows])
            teacher.observe(lv, rewards)
            step_stats["mean_reward"].append(rewards.mean())
            w = est(rewards)
            step_stats["group_k"].append(int(rewards.sum()))
            step_stats["coefficient_mass"].append(float(np.abs(w).sum()))
            if not np.any(w != 0):
                step_stats["dead_groups"] += 1
                if args.hindsight_dense:
                    # relabel every rollout whose legal prefix is deep enough:
                    # each becomes a success for the cell it reached
                    for j in rows:
                        if len(hs_prompts) >= args.hindsight_cap:
                            break
                        toks = [int(x) for x in resp[j] if int(x) != PAD]
                        n_ok, pos = simulate_prefix(task.grid, toks)
                        if n_ok >= args.hindsight_min_depth and pos != (1, 1):
                            hs_prompts.append(encode_prompt(task.grid, pos))
                            hs_resps.append(toks[:n_ok] + [EOS])
                            hs_depths.append(n_ok)
                            step_stats["relabeled"] += 1
                elif args.hindsight:
                    # relabel: goal <- deepest cell legally reached in group
                    best_n, best_pos, best_j = 0, None, None
                    for j in rows:
                        toks = [int(x) for x in resp[j] if int(x) != PAD]
                        n_ok, pos = simulate_prefix(task.grid, toks)
                        if n_ok > best_n and pos != (1, 1):
                            best_n, best_pos, best_j = n_ok, pos, j
                    if best_j is not None and best_n >= 4:
                        toks = [int(x) for x in resp[best_j] if int(x) != PAD]
                        hs_prompts.append(encode_prompt(task.grid, best_pos))
                        hs_resps.append(toks[:best_n] + [EOS])
                        hs_depths.append(best_n)
                        step_stats["relabeled"] += 1
                continue
            keep_rows.extend(rows)
            keep_w.extend(w)

        if keep_rows or hs_prompts:
            opt.zero_grad()
            if keep_rows:
                rows_t = torch.tensor(keep_rows, device=DEVICE)
                w_t = torch.tensor(np.array(keep_w), device=DEVICE, dtype=torch.float32)
                # micro-batch the backward pass to bound memory
                mb = 128
                for i in range(0, len(keep_rows), mb):
                    sel = rows_t[i:i + mb]
                    lp, _ = response_logprobs(model, prompts[sel], plens[sel], resp[sel])
                    loss = -(w_t[i:i + mb] * lp).sum() / args.tasks_per_step
                    loss.backward()
            if hs_prompts:
                # each relabeled trajectory acts as a K=1 MaxRL group:
                # w_succ = 1 - 1/N, scaled
                hp, hlens = pad_batch(hs_prompts, DEVICE)
                max_r = max(len(r) for r in hs_resps)
                hr = torch.full((len(hs_resps), max_r), PAD, dtype=torch.long,
                                device=DEVICE)
                for b, rr in enumerate(hs_resps):
                    hr[b, :len(rr)] = torch.tensor(rr, device=DEVICE)
                w_hs = args.hindsight_scale * (1.0 - 1.0 / args.rollouts)
                lp, _ = response_logprobs(model, hp, hlens, hr)
                loss = -(w_hs * lp).sum() / args.tasks_per_step
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            optimizer_steps += 1

        if args.hindsight_to_teacher and hs_depths:
            # relabeled successes nudge the matching level's posterior so the
            # curriculum advances with hindsight gains instead of waiting for
            # natural successes.  NOTE: deliberately optimistic evidence — the
            # model reached SOME cell at distance d, not a requested one; the
            # posterior decay corrects any overshoot within a few groups.
            from maze_env import LEVEL_DIST
            for d in hs_depths:
                lv_match = min(LEVELS, key=lambda l: abs(LEVEL_DIST[l] - d))
                teacher.observe(lv_match, np.array([1.0]))

        if telemetry_f is not None:
            telemetry = {
                "record_type": "telemetry",
                "protocol": args.protocol,
                "completed_updates": completed_updates,
                "selected_levels": levels,
                "group_k": step_stats["group_k"],
                "coefficient_mass": step_stats["coefficient_mass"],
                "coefficient_mass_total": float(
                    np.sum(step_stats["coefficient_mass"])),
                "dead_groups": step_stats["dead_groups"],
                "optimizer_step_applied": bool(keep_rows or hs_prompts),
                "optimizer_step": optimizer_steps,
            }
            telemetry_f.write(json.dumps(telemetry) + "\n")
            telemetry_f.flush()

        should_eval = (completed_updates in eval_schedule) if is_v2 else (
            step % args.eval_every == 0 or step == args.steps - 1)
        if should_eval:
            if is_v2:
                current_eval_generator = torch.Generator(device=DEVICE)
                current_eval_generator.manual_seed(
                    args.eval_sample_seed + completed_updates)
            else:
                current_eval_generator = None
            ev = evaluate(model, eval_tasks, n_samples=args.eval_samples,
                          generator=current_eval_generator)
            passk = ev.pop("passk")
            if is_v2:
                rec = {
                    "record_type": "evaluation",
                    "protocol": args.protocol,
                    "phase": "rl",
                    "completed_updates": completed_updates,
                    "eval": ev,
                    "passk": passk,
                    "teacher_p_hat": teacher.p_hat().round(3).tolist(),
                    "teacher_dist": teacher_distribution_snapshot(
                        teacher).round(3).tolist(),
                    "dead_groups": step_stats["dead_groups"],
                    "relabeled": step_stats["relabeled"],
                    "train_mean_reward": float(
                        np.mean(step_stats["mean_reward"])),
                    "optimizer_step": optimizer_steps,
                    "elapsed": time.time() - t0,
                }
                if completed_updates == args.steps:
                    rec["final"] = True
            else:
                rec = {"step": step, "eval": ev, "passk": passk,
                       "teacher_p_hat": teacher.p_hat().round(3).tolist(),
                       "teacher_dist": teacher.distribution().round(3).tolist(),
                       "dead_groups": step_stats["dead_groups"],
                       "relabeled": step_stats["relabeled"],
                       "train_mean_reward": float(
                           np.mean(step_stats["mean_reward"])),
                       "elapsed": time.time() - t0}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if is_v2:
                print(f"completed_updates={completed_updates} evaluation complete "
                      f"elapsed={time.time()-t0:.0f}s "
                      "(endpoints sealed in JSONL)", flush=True)
            else:
                mean_ev = np.mean(list(ev.values()))
                mean_p8 = np.mean([v.get(8, 0.0) for v in passk.values()])
                print(f"step {step:4d} mean_eval={mean_ev:.3f} "
                      f"mean_pass@8={mean_p8:.3f} "
                      f"levels={dict((k, round(v, 2)) for k, v in ev.items())} "
                      f"dead={step_stats['dead_groups']} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    # legacy_v1 historically emitted an additional final evaluation (including
    # duplicates at fixed-step termination); preserve it byte-schema-wise.
    # V2's final record was already emitted exactly once at update 250.
    if not is_v2:
        ev = evaluate(model, eval_tasks, n_samples=args.eval_samples)
        passk = ev.pop("passk")
        rec = {"step": step, "eval": ev, "passk": passk, "final": True,
               "teacher_p_hat": teacher.p_hat().round(3).tolist(),
               "elapsed": time.time() - t0}
        log_f.write(json.dumps(rec) + "\n")
        print(f"FINAL step {step} mean_eval={np.mean(list(ev.values())):.3f} "
              f"mean_pass@8={np.mean([v.get(8, 0.0) for v in passk.values()]):.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    if args.save_ckpt:
        torch.save(model.state_dict(), args.save_ckpt)
        print(f"saved checkpoint to {args.save_ckpt}", flush=True)
    log_f.close()
    if telemetry_f is not None:
        telemetry_f.close()


if __name__ == "__main__":
    main()
