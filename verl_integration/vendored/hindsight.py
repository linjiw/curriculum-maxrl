"""Exact-verifier hindsight relabeling for verl (E-LLM-2: Countdown).

Dead groups (all N rollouts scored 0) are recycled: each failed trace whose
equation is well-formed and uses the numbers exactly once evaluates to SOME
integer v — an exact success of the task (same numbers, target=v). We
rewrite the prompt's target slot to v (P6 contract 2), rebuild the row's
input_ids/attention_mask/position_ids, and set its reward to 1.

Placement contract (ray_trainer): AFTER compute_reward, BEFORE
compute_log_prob — the old-log-probs must be computed under the REWRITTEN
conditioning or the relabeled gradient is trained against stale context
(the gridworld ablation measured that mistake at -0.06 AUC, worse than no
hindsight). Requires the synchronous reward path.

Posterior hygiene (V4): the curriculum teacher must observe PRE-relabel
scores only; the trainer snapshots them before calling relabel_batch.

Group semantics: relabeled rollouts keep their uid group, so this fork's
maxrl advantage ((r−mean)/(mean+eps), mean-normalized) forms a K-of-N
contrast over the relabeled successes. NOTE the normalization makes the
advantages scale-invariant, so `scale` is a NO-OP for maxrl/grpo
estimators (adversarial review finding 4) — kept only for future
unnormalized estimators; leave at 1.0.

Mixed-target caveat (design review): each trace relabels to its own
achieved value, so one uid group can contain successes of DIFFERENT
relabeled tasks. The group contrast is then "reached some integer" vs
"reached none" rather than K-of-N of a single task — a coarser but
still-exact success signal (every relabeled trace is a true success of
its rewritten prompt; the per-trace conditioning is what the gradient
sees). The CPU reference instead picks ONE target per group; we keep
per-trace here because countdown groups rarely share achieved values.
Pre-registered as an explicit deviation (with two consequences the
design review derived: the -1/N failure weight is no longer a zero-mean
baseline but a conditional push-down on malformed outputs, and success
weights couple across unrelated targets — 1/K−1/N vs (N−1)/N for a
natural 1-of-N group). Verified: verl's normalized maxrl advantage on a
relabeled group is exactly N× the CPU maxrl_weights (same direction).
Re-examine per-trace vs one-target-per-group if E-LLM-2's hindsight
arms underperform the CPU pattern; never change one side alone.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _achieved_value(solution_str: str, numbers) -> int | None:
    """AST-whitelisted relabel map (shared semantics with
    curriculum_maxrl/countdown/countdown_reward.py — keep in sync)."""
    from curriculum_maxrl.countdown.countdown_reward import achieved_value
    return achieved_value(solution_str, numbers)


class CountdownHindsight:
    """Relabels dead Countdown groups inside a verl DataProto batch."""

    SUPPORTED_SOURCES = ("countdown", "countdown_tiered",
                         "countdown_tier0", "countdown_tier1",
                         "countdown_tier2")

    def __init__(self, tokenizer, max_prompt_length: int,
                 max_response_length: int = 1024,
                 scale: float = 1.0, max_groups_per_step: int = 8,
                 utility_gate: bool = False, gate_max_p: float = 0.5,
                 gate_decay: float = 0.9,
                 one_target_per_group: bool = False):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.scale = scale
        self.max_groups = max_groups_per_step
        # E-LLM-2b mitigation (SHARPENING_SYNTHESIS.md): the same derived
        # utility that schedules sampling gates the relabel DESTINATION.
        # A relabel to a value the model reaches often is a task at p~1,
        # where u(p)=pass@N-pass@1 says signal is zero — training there
        # sharpens the policy onto reachable outputs at coverage's expense.
        # We track a decayed Beta posterior over destination-value hit
        # rates: p_hat = how often value v appears among a group's achieved
        # values. Reject relabels with p_hat > gate_max_p.
        self.utility_gate = utility_gate
        self.gate_max_p = gate_max_p
        self.gate_decay = gate_decay
        # Draft-review 2026-08-04 P0-2 ablation arm: when True, pick ONE
        # destination per dead group (the modal achieved value across the
        # group's parseable failures; ties -> least gate-saturated, then
        # smallest) and relabel ONLY the rows that certify that value.
        # Every relabeled row in a uid group then trains toward the same
        # rewritten task, so the group contrast is K-of-N of a single
        # destination -- the object Remark 3 licenses -- instead of the
        # per-row weighted-SFT update. Rows achieving other values stay
        # failures (contrast preserved).
        self.one_target_per_group = one_target_per_group
        self._dest_hits: dict = {}    # v -> [alpha-1, beta-1] decayed counts
        self._batch_hits: set = set() # values reinforced in the current batch

    # -- helpers -----------------------------------------------------------

    # -- domain hooks (override in subclasses; defaults = Countdown) -------

    def _gt_context(self, gt) -> tuple[int, object]:
        """(old_target, domain context) from a ground_truth dict."""
        return int(gt["target"]), list(gt["numbers"])

    def _relabel_candidates(self, resp_text: str, context) -> list[int]:
        """Achieved values a failed trace certifies (may be empty)."""
        v = _achieved_value(resp_text, context)
        return [] if v is None else [int(v)]

    def _prompt_slot_pattern(self, old_target: int) -> re.Pattern:
        """Regex whose group(1)+target(+group(2) if present) is the goal
        slot in the decoded prompt."""
        return re.compile(r"(equals )" + re.escape(str(old_target)) + r"\b")

    def _dest_key(self, v: int, context):
        """Gate-posterior key for a relabel destination.

        Default: the bare value — correct for Countdown, where values
        span hundreds of integers and value ≈ task. Domains whose
        achieved values are few and shared across tasks (Jugs: ≤20
        amounts) MUST key on the full relabeled task, or the posterior
        collides unrelated tasks and saturates instantly (E-LLM-3
        postmortem: reject fraction .998, dose 0.3/step vs 169/step).
        """
        return v

    def _rewrite_prompt_ids(self, prompt_ids: torch.Tensor,
                            prompt_mask: torch.Tensor,
                            old_target: int, new_target: int):
        """Decode the (left-padded) prompt row, swap the target slot,
        re-tokenize, re-left-pad. None if the slot is missing or the new
        prompt exceeds max_prompt_length."""
        valid = prompt_ids[prompt_mask.bool()]
        text = self.tokenizer.decode(valid, skip_special_tokens=False)
        pattern = self._prompt_slot_pattern(old_target)
        ngroups = pattern.groups
        repl = r"\g<1>" + str(new_target) + (r"\g<2>" if ngroups >= 2 else "")
        new_text, n = pattern.subn(repl, text, count=1)
        if n != 1:
            return None
        new_ids = self.tokenizer(new_text, add_special_tokens=False,
                                 return_tensors="pt")["input_ids"][0]
        if len(new_ids) > self.max_prompt_length:
            return None
        pad = self.max_prompt_length - len(new_ids)
        pad_id = self.tokenizer.pad_token_id
        out_ids = torch.full((self.max_prompt_length,), pad_id,
                             dtype=prompt_ids.dtype)
        out_mask = torch.zeros(self.max_prompt_length, dtype=prompt_mask.dtype)
        out_ids[pad:] = new_ids.to(prompt_ids.dtype)
        out_mask[pad:] = 1
        return out_ids, out_mask

    def _rewrite_response_text(self, resp_text: str, old_target: int,
                               v: int) -> str:
        """Keep the certified response byte-for-byte unchanged.

        The clean SFT prerequisite deliberately uses target-agnostic reasoning
        text, so only the prompt's explicit target slot needs rewriting.  A
        blanket number substitution can silently alter decimals or intermediate
        arithmetic and is therefore not semantically safe.
        """
        return resp_text

    def _rewrite_response_ids(self, new_text: str, block_len: int):
        """Tokenize rewritten response, right-pad to the batch's actual
        response block length. None if it no longer fits."""
        ids = self.tokenizer(new_text, add_special_tokens=False,
                             return_tensors="pt")["input_ids"][0]
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            ids = torch.cat([ids, torch.tensor([eos], dtype=ids.dtype)])
        if len(ids) > block_len:
            return None
        pad_id = self.tokenizer.pad_token_id
        out = torch.full((block_len,), pad_id, dtype=torch.long)
        mask = torch.zeros(block_len, dtype=torch.long)
        out[:len(ids)] = ids
        mask[:len(ids)] = 1
        return out, mask

    # -- main entry --------------------------------------------------------

    def relabel_batch(self, batch, reward_tensor: torch.Tensor) -> dict:
        """Mutates batch tensors + reward_tensor in place for relabeled rows.

        Returns metrics. Must run before compute_log_prob (see module doc).
        """
        stats = {"hindsight/dead_groups": 0, "hindsight/relabeled_groups": 0,
                 "hindsight/relabeled_rollouts": 0, "hindsight/skipped_rewrite": 0,
                 "hindsight/aux_group_response_tokens": 0,
                 "_accepted_group_token_counts": [],
                 "_accepted_groups": []}
        ntb = batch.non_tensor_batch
        if "uid" not in ntb or "reward_model" not in ntb:
            return stats
        data_sources = ntb.get("data_source")
        scores = reward_tensor.sum(dim=-1)
        stats["hindsight/pre_optimizer_response_tokens_total"] = int(
            batch.batch["response_mask"].sum().item())

        # group rows by uid; a group is dead iff every rollout scored 0
        rows_by_uid = defaultdict(list)
        for i, uid in enumerate(ntb["uid"]):
            rows_by_uid[uid].append(i)
        dead_uids = []
        for uid, rows in rows_by_uid.items():
            if data_sources is not None and \
                    data_sources[rows[0]] not in self.SUPPORTED_SOURCES:
                continue
            if float(scores[rows].sum()) == 0.0:
                dead_uids.append(uid)
        stats["hindsight/dead_groups"] = len(dead_uids)
        if not dead_uids:
            return stats

        prompt_len = batch.batch["prompts"].shape[1]
        response_mask = batch.batch["response_mask"]
        relabeled_groups = 0
        for uid in dead_uids[: self.max_groups]:
            rows = rows_by_uid[uid]
            gt = ntb["reward_model"][rows[0]]["ground_truth"]
            old_target, context = self._gt_context(gt)
            group_hit = False
            group_target = None
            if self.one_target_per_group:
                # pass 1: modal achieved value across the group's parseable
                # failures; ties -> least gate-saturated, then smallest
                counts: dict = {}
                for i in rows:
                    resp_ids = batch.batch["responses"][i]
                    txt = self.tokenizer.decode(
                        resp_ids[response_mask[i].bool()],
                        skip_special_tokens=True)
                    for c in self._relabel_candidates(txt, context):
                        if c != old_target:
                            counts[c] = counts.get(c, 0) + 1
                if not counts:
                    continue

                def _sat(c):
                    a, b = self._dest_hits.get(
                        self._dest_key(c, context), (0.0, 0.0))
                    return (a + 1.0) / (a + b + 2.0)
                group_target = min(counts,
                                   key=lambda c: (-counts[c], _sat(c), c))
                if self.utility_gate:
                    # gate once per GROUP (one destination -> one decision;
                    # per-row checks would record duplicate hits for the
                    # same key and inflate the posterior)
                    key = self._dest_key(group_target, context)
                    a, b = self._dest_hits.get(key, (0.0, 0.0))
                    p_hat = (a + 1.0) / (a + b + 2.0)
                    self._dest_hits[key] = (a * self.gate_decay + 1.0,
                                            b * self.gate_decay)
                    self._batch_hits.add(key)
                    if p_hat > self.gate_max_p:
                        stats["hindsight/gated_saturated"] = \
                            stats.get("hindsight/gated_saturated", 0) + 1
                        continue
            for i in rows:
                resp_ids = batch.batch["responses"][i]
                resp_text = self.tokenizer.decode(
                    resp_ids[response_mask[i].bool()], skip_special_tokens=True)
                cands = [c for c in self._relabel_candidates(resp_text, context)
                         if c != old_target]
                if self.one_target_per_group:
                    # only rows that certify the group's single destination
                    cands = [c for c in cands if c == group_target]
                if not cands:
                    continue
                # least-saturated candidate first when gating (the gate's
                # own posterior orders them); else first (Countdown yields
                # exactly one anyway)
                if self.utility_gate and len(cands) > 1:
                    def p_hat_of(c):
                        a, b = self._dest_hits.get(
                            self._dest_key(c, context), (0.0, 0.0))
                        return (a + 1.0) / (a + b + 2.0)
                    cands.sort(key=p_hat_of)
                v = cands[0]
                if self.utility_gate and not self.one_target_per_group:
                    # per-row gate (one-target mode gates once per group,
                    # above, to avoid duplicate posterior hits per key)
                    key = self._dest_key(v, context)
                    a, b = self._dest_hits.get(key, (0.0, 0.0))
                    p_hat = (a + 1.0) / (a + b + 2.0)   # Beta(1,1) prior
                    # record the observation regardless of admission
                    self._dest_hits[key] = (a * self.gate_decay + 1.0,
                                            b * self.gate_decay)
                    self._batch_hits.add(key)
                    if p_hat > self.gate_max_p:
                        stats["hindsight/gated_saturated"] = \
                            stats.get("hindsight/gated_saturated", 0) + 1
                        continue
                rewritten = self._rewrite_prompt_ids(
                    batch.batch["prompts"][i],
                    batch.batch["attention_mask"][i, :prompt_len],
                    old_target, v)
                if rewritten is None:
                    stats["hindsight/skipped_rewrite"] += 1
                    continue
                new_prompt_ids, new_prompt_mask = rewritten
                new_resp = self._rewrite_response_text(resp_text,
                                                       old_target, v)
                resp_rewrite = self._rewrite_response_ids(
                    new_resp, batch.batch["responses"].shape[1])
                if resp_rewrite is None:
                    stats["hindsight/skipped_rewrite"] += 1
                    continue
                new_resp_ids, new_resp_mask = resp_rewrite
                device = batch.batch["input_ids"].device
                batch.batch["prompts"][i] = new_prompt_ids.to(device)
                batch.batch["input_ids"][i, :prompt_len] = new_prompt_ids.to(device)
                batch.batch["attention_mask"][i, :prompt_len] = new_prompt_mask.to(device)
                # left-padded rows: position_ids = clip(cumsum(mask)-1, 0).
                # Only valid for 2-D rope positions; 3-D mrope (VLMs) would
                # need per-section recompute — refuse loudly rather than
                # corrupt silently (adversarial review finding 7).
                assert batch.batch["position_ids"].dim() == 2, \
                    "hindsight relabeling supports 2-D position_ids only"
                row_mask = batch.batch["attention_mask"][i]
                batch.batch["position_ids"][i] = torch.clip(
                    torch.cumsum(row_mask, dim=-1) - 1, min=0)
                # Re-tokenize the unchanged certified response into the
                # right-padded response region.
                batch.batch["responses"][i] = new_resp_ids.to(device)
                batch.batch["input_ids"][i, prompt_len:] = new_resp_ids.to(device)
                batch.batch["attention_mask"][i, prompt_len:] = new_resp_mask.to(device)
                batch.batch["response_mask"][i] = new_resp_mask.to(device)
                row_mask = batch.batch["attention_mask"][i]
                batch.batch["position_ids"][i] = torch.clip(
                    torch.cumsum(row_mask, dim=-1) - 1, min=0)
                # reward: 1*scale at the last valid (rewritten) response token
                valid_idx = new_resp_mask.nonzero()
                if len(valid_idx) == 0:
                    continue
                reward_tensor[i, :] = 0.0
                reward_tensor[i, valid_idx[-1]] = self.scale
                stats["hindsight/relabeled_rollouts"] += 1
                group_hit = True
            if group_hit:
                rollout_tokens = [int(value) for value in
                                  response_mask[rows].sum(dim=-1).tolist()]
                group_tokens = sum(rollout_tokens)
                stats["_accepted_group_token_counts"].append(group_tokens)
                dataset_index = None
                if "index" in ntb:
                    dataset_index = ntb["index"][rows[0]]
                    if hasattr(dataset_index, "item"):
                        dataset_index = dataset_index.item()
                stats["_accepted_groups"].append({
                    "data_source": str(data_sources[rows[0]])
                    if data_sources is not None else None,
                    "dataset_index": dataset_index,
                    "response_tokens": group_tokens,
                    "rollout_token_counts": rollout_tokens,
                    "rollouts": len(rows),
                })
                stats["hindsight/aux_group_response_tokens"] += group_tokens
            relabeled_groups += int(group_hit)
        if self.utility_gate:
            # values NOT reinforced this batch decay toward the prior on
            # BOTH sides (audit F5: the old loop decayed everything incl.
            # just-hit values via a one-sided pseudo-miss, under-gating
            # systematically, and its prune was unreachable)
            for key in list(self._dest_hits):
                if key in self._batch_hits:
                    continue
                a, b = self._dest_hits[key]
                self._dest_hits[key] = (a * self.gate_decay,
                                        b * self.gate_decay)
                if a + b < 0.05:
                    del self._dest_hits[key]
            self._batch_hits.clear()
            stats["hindsight/gate_tracked_values"] = len(self._dest_hits)
        stats["hindsight/relabeled_groups"] = relabeled_groups
        stats["hindsight/optimizer_rows_total"] = len(batch)
        stats["hindsight/optimizer_response_tokens_total"] = int(
            batch.batch["response_mask"].sum().item())
        attempted = sum(len(rows_by_uid[u]) for u in dead_uids[: self.max_groups])
        if attempted:
            stats["hindsight/relabel_yield"] = (
                stats["hindsight/relabeled_rollouts"] / attempted)
        return stats


class DoseMatchedLiveReplay:
    """Replay informative live groups in B2's accepted prompt-group slots.

    A source B2 audit fixes the number and response-token budget of auxiliary
    groups at every step, including the dataset indices of the slots that B2
    accepted. The replay arm replaces those exact slots with copies of
    informative live groups from its current generated batch. A scheduled slot
    can become informative after the policies diverge; displacing such a group
    is unavoidable in a fixed-size on-policy batch and is audited explicitly.
    No new generations or relabel information are used. An optional bounded
    buffer can retain prior informative groups for batches with no current
    live source; the source age is audited explicitly. Full groups are replayed,
    so exact token equality is discrete; cumulative auxiliary and total-
    optimizer token ledgers turn that limitation into an automatic treatment-
    delivery check rather than a hidden approximation.
    """

    SUPPORTED_SOURCES = CountdownHindsight.SUPPORTED_SOURCES

    def __init__(self, schedule_path: str, seed: int = 1,
                 max_cumulative_token_mismatch_fraction: float = 0.05,
                 strict: bool = True, buffer_capacity_groups: int = 0,
                 max_buffer_age_steps: int = 8,
                 reservoir_path: str | None = None,
                 reservoir_sha256: str | None = None):
        self.schedule_path = Path(schedule_path)
        self.seed = int(seed)
        self.max_mismatch_fraction = float(
            max_cumulative_token_mismatch_fraction)
        self.strict = bool(strict)
        self.buffer_capacity_groups = int(buffer_capacity_groups)
        if self.buffer_capacity_groups < 0:
            raise ValueError("buffer_capacity_groups must be nonnegative")
        self.max_buffer_age_steps = int(max_buffer_age_steps)
        if self.max_buffer_age_steps < 1:
            raise ValueError("max_buffer_age_steps must be positive")
        self.replay_buffer = []
        self.reservoir_path = (Path(reservoir_path)
                               if reservoir_path else None)
        self.reservoir_sha256 = None
        self.reservoir_groups = []
        if self.reservoir_path is not None:
            if not self.reservoir_path.is_file():
                raise ValueError(
                    f"replay reservoir does not exist: {self.reservoir_path}")
            digest = self._sha256(self.reservoir_path)
            if (reservoir_sha256 is not None and
                    digest.lower() != str(reservoir_sha256).lower()):
                raise ValueError(
                    f"replay reservoir SHA-256 {digest} != expected "
                    f"{reservoir_sha256}")
            try:
                artifact = torch.load(
                    self.reservoir_path, map_location="cpu",
                    weights_only=False)
            except TypeError:  # PyTorch before weights_only was introduced.
                artifact = torch.load(self.reservoir_path, map_location="cpu")
            if artifact.get("format_version") != 1:
                raise ValueError("unsupported replay reservoir format")
            for position, group in enumerate(artifact.get("groups", [])):
                if group.get("status") != "informative":
                    raise ValueError(
                        f"reservoir group {position} is not informative")
                copied = copy.deepcopy(group)
                copied["uid"] = f"reservoir:{position}"
                copied["source_kind"] = "reservoir"
                copied["source_step"] = None
                copied["rows"] = list(range(int(copied["group_size"])))
                self.reservoir_groups.append(copied)
            if not self.reservoir_groups:
                raise ValueError("empty replay reservoir")
            self.reservoir_sha256 = digest
        self.cumulative_target_tokens = 0
        self.cumulative_replay_tokens = 0
        self.cumulative_target_optimizer_tokens = 0
        self.cumulative_optimizer_tokens = 0
        self.schedule = {}
        with self.schedule_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                step = int(row["global_step"])
                if step in self.schedule:
                    raise ValueError(f"duplicate replay schedule step {step}")
                self.schedule[step] = row
        if not self.schedule:
            raise ValueError(f"empty replay schedule: {self.schedule_path}")

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _dataset_index(ntb, rows):
        if "index" not in ntb:
            return None
        value = ntb["index"][rows[0]]
        return value.item() if hasattr(value, "item") else value

    @staticmethod
    def _snapshot_non_tensor_rows(values, source_rows):
        if isinstance(values, np.ndarray):
            return values[source_rows].copy()
        return [copy.deepcopy(values[index]) for index in source_rows]

    @staticmethod
    def _write_non_tensor_rows(values, target_rows, copied):
        if isinstance(values, np.ndarray):
            values[target_rows] = copied
            return
        for target, value in zip(target_rows, copied):
            values[target] = value

    def _snapshot_group(self, batch, reward_tensor, ntb, rows):
        return {
            "batch": {key: tensor[rows].clone()
                      for key, tensor in batch.batch.items()},
            "reward": reward_tensor[rows].clone(),
            "non_tensor": {
                key: self._snapshot_non_tensor_rows(values, rows)
                for key, values in ntb.items() if key != "uid"
            },
        }

    def _select_sources(self, live_groups, target_groups,
                        target_aux_tokens_after_step: int,
                        target_optimizer_tokens_after_step: int,
                        pre_optimizer_tokens: int, global_step: int):
        """Jointly length-match auxiliary and total optimizer token dose."""
        if not target_groups:
            return []
        rng = np.random.default_rng(self.seed * 1_000_003 +
                                    int(global_step) * 101 +
                                    target_aux_tokens_after_step)
        order = list(range(len(live_groups)))
        rng.shuffle(order)
        # Response blocks are short, so attainable token sums stay small even
        # with replacement. Retain indices because a slot cannot replay itself.
        states = {0: []}
        for target in target_groups:
            # With replacement, two eligible sources with the same aggregate
            # token count induce identical DP states. The old loop visited all
            # of them and ``setdefault`` retained the first in seeded order;
            # deduplicating to that same first representative preserves the
            # exact selection while avoiding O(reservoir_size) duplicate work
            # in every state transition.
            eligible = []
            seen_token_counts = set()
            for group_index in order:
                source = live_groups[group_index]
                if source["uid"] == target["uid"]:
                    continue
                if (source["source_kind"] == "reservoir" and
                        source["dataset_index"] ==
                        target["dataset_index"]):
                    continue
                if source["tokens"] in seen_token_counts:
                    continue
                seen_token_counts.add(source["tokens"])
                eligible.append(group_index)
            next_states = {}
            for total, selected in states.items():
                for group_index in eligible:
                    source = live_groups[group_index]
                    new_total = total + source["tokens"]
                    next_states.setdefault(new_total, selected + [group_index])
            if not next_states:
                raise RuntimeError(
                    "no informative replay source distinct from its target "
                    f"slot at step {global_step}")
            states = next_states

        target_current_tokens = sum(group["tokens"] for group in target_groups)

        def objective(total):
            aux_delta = (self.cumulative_replay_tokens + total -
                         target_aux_tokens_after_step)
            optimizer_step_tokens = (pre_optimizer_tokens -
                                     target_current_tokens + total)
            optimizer_delta = (self.cumulative_optimizer_tokens +
                               optimizer_step_tokens -
                               target_optimizer_tokens_after_step)
            aux_fraction = abs(aux_delta) / max(target_aux_tokens_after_step, 1)
            optimizer_fraction = (abs(optimizer_delta) /
                                  max(target_optimizer_tokens_after_step, 1))
            return (max(aux_fraction, optimizer_fraction),
                    aux_fraction + optimizer_fraction,
                    abs(aux_delta), abs(optimizer_delta), total)

        best_total = min(states, key=objective)
        return [live_groups[index] for index in states[best_total]]

    def replay_batch(self, batch, reward_tensor: torch.Tensor,
                     global_step: int) -> dict:
        if global_step not in self.schedule:
            raise ValueError(f"replay schedule has no step {global_step}")
        schedule_row = self.schedule[global_step]
        accepted_groups = schedule_row.get("accepted_groups", [])
        target_counts = [int(group["response_tokens"])
                         for group in accepted_groups]
        if not target_counts:
            target_counts = [int(value) for value in
                             schedule_row.get("accepted_group_token_counts", [])]
        target_count = len(target_counts)
        target_step_tokens = sum(target_counts)

        ntb = batch.non_tensor_batch
        if "uid" not in ntb:
            raise ValueError("live replay requires uid groups")
        data_sources = ntb.get("data_source")
        response_mask = batch.batch["response_mask"]
        scores = reward_tensor.sum(dim=-1)
        pre_success_rollouts = int((scores > 0).sum().item())
        pre_optimizer_tokens = int(response_mask.sum().item())
        rows_by_uid = defaultdict(list)
        for row_index, uid in enumerate(ntb["uid"]):
            rows_by_uid[uid].append(row_index)

        all_groups = []
        dead_groups = []
        saturated_groups = []
        live_groups = []
        for uid, rows in rows_by_uid.items():
            if (data_sources is not None and
                    data_sources[rows[0]] not in self.SUPPORTED_SOURCES):
                continue
            group_scores = scores[rows]
            info = {
                "uid": uid,
                "rows": rows,
                "tokens": int(response_mask[rows].sum().item()),
                "dataset_index": self._dataset_index(ntb, rows),
                "data_source": str(data_sources[rows[0]])
                if data_sources is not None else None,
            }
            group_min = float(group_scores.min().item())
            group_max = float(group_scores.max().item())
            if group_min == group_max:
                if group_max == 0.0:
                    info["status"] = "dead"
                    dead_groups.append(info)
                else:
                    info["status"] = "saturated"
                    saturated_groups.append(info)
            else:
                info["status"] = "informative"
                info["source_kind"] = "current"
                info["source_step"] = int(global_step)
                info["payload"] = self._snapshot_group(
                    batch, reward_tensor, ntb, rows)
                live_groups.append(info)
            all_groups.append(info)

        buffer_before_age_filter = len(self.replay_buffer)
        buffer_candidates = [
            group for group in self.replay_buffer
            if int(global_step) - group["source_step"] <=
            self.max_buffer_age_steps
        ]
        buffer_evicted_for_age = buffer_before_age_filter - len(
            buffer_candidates)
        buffer_candidate_count = len(buffer_candidates)
        self.replay_buffer = buffer_candidates
        # A frozen reservoir defines a distinct E2c treatment: never mix its
        # off-policy sources with current-policy or recent-buffer sources.
        source_candidates = (self.reservoir_groups
                             if self.reservoir_groups
                             else live_groups + buffer_candidates)
        if target_count and not source_candidates:
            raise RuntimeError(
                f"step {global_step}: B2 requests replay but no informative "
                "current or buffered live group exists")

        groups_by_index = {group["dataset_index"]: group
                           for group in all_groups}
        target_groups = []
        fallback_slots = 0
        for position, target_tokens in enumerate(target_counts):
            scheduled_index = (accepted_groups[position].get("dataset_index")
                               if position < len(accepted_groups) else None)
            chosen = groups_by_index.get(scheduled_index)
            if chosen is None:
                if self.strict:
                    raise RuntimeError(
                        f"step {global_step}: B2 accepted dataset index "
                        f"{scheduled_index!r}, which is absent from the "
                        "matched generated batch")
                fallback_slots += 1
                unused = [group for group in all_groups
                          if group not in target_groups]
                if not unused:
                    raise RuntimeError(
                        f"step {global_step}: no unused replay target slot")
                chosen = min(unused,
                             key=lambda group: abs(group["tokens"] -
                                                   target_tokens))
            if chosen in target_groups:
                raise RuntimeError(
                    f"step {global_step}: duplicate replay target slot "
                    f"{chosen['dataset_index']!r}")
            target_groups.append(chosen)

        expected_optimizer_tokens = int(schedule_row.get(
            "hindsight/optimizer_response_tokens_total",
            pre_optimizer_tokens))
        target_aux_tokens_after_step = (self.cumulative_target_tokens +
                                        target_step_tokens)
        target_optimizer_tokens_after_step = (
            self.cumulative_target_optimizer_tokens +
            expected_optimizer_tokens)
        source_groups = self._select_sources(
            source_candidates, target_groups, target_aux_tokens_after_step,
            target_optimizer_tokens_after_step, pre_optimizer_tokens,
            global_step)

        # Current-group payloads were snapshotted before any write, so sources
        # that are also different target slots cannot be corrupted by copy
        # order. Buffered payloads are already immutable snapshots.
        source_payloads = [source["payload"] for source in source_groups]

        if self.buffer_capacity_groups:
            self.replay_buffer.extend(
                [{**group, "source_kind": "buffer"}
                 for group in live_groups])
            self.replay_buffer = self.replay_buffer[
                -self.buffer_capacity_groups:]

        replay_records = []
        replay_step_tokens = 0
        displaced_live_slots = 0
        for target_tokens, target, source, payload in zip(
                target_counts, target_groups, source_groups, source_payloads):
            if len(target["rows"]) != len(source["rows"]):
                raise RuntimeError("replay source and target group sizes differ")
            target_rows = target["rows"]
            for key, tensor in batch.batch.items():
                tensor[target_rows] = payload["batch"][key].to(tensor.device)
            reward_tensor[target_rows] = payload["reward"].to(
                reward_tensor.device)
            # Preserve target uid: the copy remains a distinct N-way group.
            for key, copied in payload["non_tensor"].items():
                self._write_non_tensor_rows(ntb[key], target_rows, copied)
            replay_tokens = int(
                batch.batch["response_mask"][target_rows].sum().item())
            replay_step_tokens += replay_tokens
            displaced_live_slots += int(target["status"] == "informative")
            replay_records.append({
                "target_response_tokens": target_tokens,
                "replay_response_tokens": replay_tokens,
                "source_dataset_index": source["dataset_index"],
                "source_data_source": source["data_source"],
                "source_kind": source["source_kind"],
                "source_step": source["source_step"],
                "source_age_steps": (None if source["source_step"] is None
                                     else int(global_step) -
                                     source["source_step"]),
                "replaced_dataset_index": target["dataset_index"],
                "replaced_data_source": target["data_source"],
                "replaced_group_status": target["status"],
            })

        self.cumulative_target_tokens += target_step_tokens
        self.cumulative_replay_tokens += replay_step_tokens
        mismatch = self.cumulative_replay_tokens - self.cumulative_target_tokens
        mismatch_fraction = (abs(mismatch) /
                             max(self.cumulative_target_tokens, 1))
        if (self.strict and target_count and
                mismatch_fraction > self.max_mismatch_fraction):
            raise RuntimeError(
                f"step {global_step}: cumulative replay-token mismatch "
                f"{mismatch_fraction:.4%} exceeds "
                f"{self.max_mismatch_fraction:.4%}")

        expected_rows = schedule_row.get(
            "hindsight/optimizer_rows_total", len(batch))
        if self.strict and int(expected_rows) != len(batch):
            raise RuntimeError(
                f"step {global_step}: optimizer rows {len(batch)} != B2 "
                f"schedule {expected_rows}")
        actual_optimizer_tokens = int(
            batch.batch["response_mask"].sum().item())
        self.cumulative_target_optimizer_tokens += expected_optimizer_tokens
        self.cumulative_optimizer_tokens += actual_optimizer_tokens
        optimizer_mismatch = (self.cumulative_optimizer_tokens -
                              self.cumulative_target_optimizer_tokens)
        optimizer_mismatch_fraction = (
            abs(optimizer_mismatch) /
            max(self.cumulative_target_optimizer_tokens, 1))
        if (self.strict and
                optimizer_mismatch_fraction > self.max_mismatch_fraction):
            raise RuntimeError(
                f"step {global_step}: cumulative optimizer-token mismatch "
                f"{optimizer_mismatch_fraction:.4%} exceeds "
                f"{self.max_mismatch_fraction:.4%}")

        return {
            "replay/groups": target_count,
            "replay/live_candidates": len(source_candidates),
            "replay/current_live_candidates": len(live_groups),
            "replay/buffer_candidates": buffer_candidate_count,
            "replay/reservoir_candidates": len(self.reservoir_groups),
            "replay/buffer_evicted_for_age": buffer_evicted_for_age,
            "replay/buffer_groups_after_step": len(self.replay_buffer),
            "replay/buffer_sources_used": sum(
                source["source_kind"] == "buffer"
                for source in source_groups),
            "replay/reservoir_sources_used": sum(
                source["source_kind"] == "reservoir"
                for source in source_groups),
            "replay/max_source_age_steps": max(
                [int(global_step) - source["source_step"]
                 for source in source_groups
                 if source["source_step"] is not None], default=0),
            "replay/dead_slots": len(dead_groups),
            "replay/saturated_slots": len(saturated_groups),
            "replay/inactive_slots": len(dead_groups) + len(saturated_groups),
            "replay/displaced_live_slots": displaced_live_slots,
            "replay/pre_success_rollouts": pre_success_rollouts,
            "replay/post_success_rollouts": int(
                (reward_tensor.sum(dim=-1) > 0).sum().item()),
            "replay/fallback_slots": fallback_slots,
            "replay/target_aux_response_tokens": target_step_tokens,
            "replay/aux_response_tokens": replay_step_tokens,
            "replay/cumulative_target_aux_response_tokens":
                self.cumulative_target_tokens,
            "replay/cumulative_aux_response_tokens":
                self.cumulative_replay_tokens,
            "replay/cumulative_token_delta": mismatch,
            "replay/cumulative_token_mismatch_fraction": mismatch_fraction,
            "replay/optimizer_rows_total": len(batch),
            "replay/pre_optimizer_response_tokens_total":
                pre_optimizer_tokens,
            "replay/optimizer_response_tokens_total": actual_optimizer_tokens,
            "replay/target_optimizer_response_tokens_total":
                expected_optimizer_tokens,
            "replay/optimizer_response_token_delta":
                actual_optimizer_tokens - expected_optimizer_tokens,
            "replay/cumulative_target_optimizer_response_tokens":
                self.cumulative_target_optimizer_tokens,
            "replay/cumulative_optimizer_response_tokens":
                self.cumulative_optimizer_tokens,
            "replay/cumulative_optimizer_token_delta": optimizer_mismatch,
            "replay/cumulative_optimizer_token_mismatch_fraction":
                optimizer_mismatch_fraction,
            "_replay_groups": replay_records,
        }


class ReplayReservoirCollector:
    """Collect immutable informative groups from a frozen-policy rollout run."""

    SUPPORTED_SOURCES = CountdownHindsight.SUPPORTED_SOURCES

    def __init__(self, output_path: str, seed: int, max_groups: int = 256,
                 expected_group_size: int = 16):
        self.output_path = Path(output_path)
        self.manifest_path = self.output_path.with_suffix(
            self.output_path.suffix + ".manifest.json")
        self.seed = int(seed)
        self.max_groups = int(max_groups)
        self.expected_group_size = int(expected_group_size)
        if self.max_groups < 1:
            raise ValueError("reservoir max_groups must be positive")
        if self.expected_group_size < 2:
            raise ValueError("reservoir group size must be at least two")
        if self.output_path.exists() or self.manifest_path.exists():
            raise ValueError(
                f"refusing to modify existing reservoir: {self.output_path}")
        self.groups = []
        self.dataset_indices = set()
        self.steps_seen = []

    @staticmethod
    def _normalize(value):
        if hasattr(value, "item") and not isinstance(value, torch.Tensor):
            try:
                value = value.item()
            except ValueError:
                pass
        if isinstance(value, np.ndarray):
            return [ReplayReservoirCollector._normalize(item)
                    for item in value.tolist()]
        if isinstance(value, dict):
            return {str(key): ReplayReservoirCollector._normalize(item)
                    for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ReplayReservoirCollector._normalize(item)
                    for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return copy.deepcopy(value)

    @staticmethod
    def _non_tensor_rows(values, rows):
        return [ReplayReservoirCollector._normalize(values[index])
                for index in rows]

    def _snapshot_group(self, batch, reward_tensor, ntb, rows):
        return {
            "batch": {key: tensor[rows].detach().cpu().clone()
                      for key, tensor in batch.batch.items()},
            "reward": reward_tensor[rows].detach().cpu().clone(),
            "non_tensor": {
                key: self._non_tensor_rows(values, rows)
                for key, values in ntb.items() if key != "uid"
            },
        }

    def _write_artifact(self):
        artifact = {
            "format_version": 1,
            "collector_seed": self.seed,
            "expected_group_size": self.expected_group_size,
            "max_groups": self.max_groups,
            "steps_seen": self.steps_seen,
            "groups": self.groups,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.output_path.with_suffix(
            self.output_path.suffix + ".tmp")
        torch.save(artifact, temporary_path)
        temporary_path.replace(self.output_path)
        digest = DoseMatchedLiveReplay._sha256(self.output_path)
        manifest = {
            "format_version": 1,
            "collector_seed": self.seed,
            "expected_group_size": self.expected_group_size,
            "max_groups": self.max_groups,
            "groups": len(self.groups),
            "unique_dataset_indices": len(self.dataset_indices),
            "distinct_response_token_counts": len({
                group["tokens"] for group in self.groups}),
            "response_token_min": min(
                (group["tokens"] for group in self.groups), default=None),
            "response_token_max": max(
                (group["tokens"] for group in self.groups), default=None),
            "steps_seen": self.steps_seen,
            "sha256": digest,
            "groups_summary": [{
                "dataset_index": group["dataset_index"],
                "data_source": group["data_source"],
                "group_size": group["group_size"],
                "response_tokens": group["tokens"],
                "success_rollouts": group["success_rollouts"],
                "source_step": group["collection_step"],
            } for group in self.groups],
        }
        temporary_manifest = self.manifest_path.with_suffix(
            self.manifest_path.suffix + ".tmp")
        with temporary_manifest.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_manifest.replace(self.manifest_path)

    def collect_batch(self, batch, reward_tensor: torch.Tensor,
                      global_step: int) -> dict:
        ntb = batch.non_tensor_batch
        if "uid" not in ntb:
            raise ValueError("reservoir collection requires uid groups")
        response_mask = batch.batch["response_mask"]
        scores = reward_tensor.sum(dim=-1)
        rows_by_uid = defaultdict(list)
        for row_index, uid in enumerate(ntb["uid"]):
            rows_by_uid[uid].append(row_index)
        data_sources = ntb.get("data_source")
        retained_before = len(self.groups)
        informative_seen = 0
        self.steps_seen.append(int(global_step))
        for _, rows in rows_by_uid.items():
            if len(self.groups) >= self.max_groups:
                break
            if len(rows) != self.expected_group_size:
                raise RuntimeError(
                    f"step {global_step}: group size {len(rows)} != "
                    f"{self.expected_group_size}")
            data_source = (str(data_sources[rows[0]])
                           if data_sources is not None else None)
            if (data_source is not None and
                    data_source not in self.SUPPORTED_SOURCES):
                continue
            group_scores = scores[rows]
            success_rollouts = int((group_scores > 0).sum().item())
            if not 0 < success_rollouts < len(rows):
                continue
            informative_seen += 1
            dataset_index = DoseMatchedLiveReplay._dataset_index(ntb, rows)
            if dataset_index is None:
                raise ValueError("reservoir collection requires dataset index")
            dataset_index = int(dataset_index)
            if dataset_index in self.dataset_indices:
                continue
            self.dataset_indices.add(dataset_index)
            self.groups.append({
                "status": "informative",
                "collection_step": int(global_step),
                "dataset_index": dataset_index,
                "data_source": data_source,
                "group_size": len(rows),
                "tokens": int(response_mask[rows].sum().item()),
                "success_rollouts": success_rollouts,
                "payload": self._snapshot_group(
                    batch, reward_tensor, ntb, rows),
            })
        self._write_artifact()
        return {
            "reservoir/groups_retained": len(self.groups),
            "reservoir/groups_added": len(self.groups) - retained_before,
            "reservoir/informative_groups_seen": informative_seen,
            "reservoir/distinct_response_token_counts": len({
                group["tokens"] for group in self.groups}),
            "reservoir/full": int(len(self.groups) >= self.max_groups),
        }


class JugsHindsight(CountdownHindsight):
    """Relabels dead Jugs groups (E-LLM-3).

    Domain differences from Countdown, all via the hooks:
    - relabel candidates = amounts in the FINAL simulated state of a
      parsed move sequence (possibly several per trace; the gate's
      posterior picks the least-saturated when gating, else the first).
    - prompt goal slot: "contain exactly {t} litres".
    - response rewrite: Jugs retains the historical word-boundary target
      swap; its <answer> move list contains no numbers (moves are `fill A`
      / `pour A->B`). Countdown's new clean SFT path instead leaves the
      certified response unchanged.

    Design note (pre-registered deviation, DESIGN_E_LLM3.md): v1 uses
    final-state amounts only — the response text is a verified success
    of the relabeled task AS-IS. The richer prefix map (any amount ever
    held, requires truncating the response) is reserved for a v2
    ablation.
    """

    SUPPORTED_SOURCES = ("jugs", "jugs_tier0", "jugs_tier1", "jugs_tier2",
                         "jugs_tier3", "jugs_tier4")

    def _gt_context(self, gt):
        return int(gt["target"]), list(gt["jug_capacities"])

    def _relabel_candidates(self, resp_text, context):
        from curriculum_maxrl.jugs.jugs_reward import achieved_amounts
        return achieved_amounts(resp_text, context)

    def _prompt_slot_pattern(self, old_target):
        return re.compile(r"(contain exactly )" + re.escape(str(old_target))
                          + r"( litres)")

    def _rewrite_response_text(self, resp_text, old_target, v):
        return re.sub(r"\b" + re.escape(str(old_target)) + r"\b",
                      str(v), resp_text)

    def _dest_key(self, v, context):
        # E-LLM-3 postmortem fix: Jugs amounts are <=20 shared integers;
        # key the gate posterior on the relabeled TASK (caps, target).
        return (tuple(sorted(context)), int(v))


HINDSIGHT_CLASSES = {
    "countdown": CountdownHindsight,
    "jugs": JugsHindsight,
}
