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

import re
from collections import defaultdict

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
        """Opus5 review B2 (v2 — repair, not discard): the SFT'd model
        states the goal in its think-trace ("I need to reach 99...").
        A relabeled pair whose response names the OLD target trains
        goal-incoherent reasoning; the v1 gate that skipped such traces
        killed ~100% of yield (the trace ALWAYS names the goal). Instead
        rewrite the response: substitute the old target for v at word
        boundaries — coherent by construction, since the trace's own
        arithmetic produces v. The expression inside <answer> is
        unaffected (it contains pool numbers only; v==old_target was
        excluded upstream, and old_target is never a pool number by pool
        construction)."""
        return re.sub(r"\b" + re.escape(str(old_target)) + r"\b",
                      str(v), resp_text)

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
                 "hindsight/relabeled_rollouts": 0, "hindsight/skipped_rewrite": 0}
        ntb = batch.non_tensor_batch
        if "uid" not in ntb or "reward_model" not in ntb:
            return stats
        data_sources = ntb.get("data_source")
        scores = reward_tensor.sum(dim=-1)

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
                # apply the rewritten response (right-padded region)
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
        attempted = sum(len(rows_by_uid[u]) for u in dead_uids[: self.max_groups])
        if attempted:
            stats["hindsight/relabel_yield"] = (
                stats["hindsight/relabeled_rollouts"] / attempted)
        return stats


class JugsHindsight(CountdownHindsight):
    """Relabels dead Jugs groups (E-LLM-3).

    Domain differences from Countdown, all via the hooks:
    - relabel candidates = amounts in the FINAL simulated state of a
      parsed move sequence (possibly several per trace; the gate's
      posterior picks the least-saturated when gating, else the first).
    - prompt goal slot: "contain exactly {t} litres".
    - response rewrite: same word-boundary target swap; the <answer>
      move list contains no numbers, so it is untouched by construction
      (moves are `fill A` / `pour A->B`), and the think-text swap keeps
      goal coherence exactly as in Countdown.

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

    def _dest_key(self, v, context):
        # E-LLM-3 postmortem fix: Jugs amounts are <=20 shared integers;
        # key the gate posterior on the relabeled TASK (caps, target).
        return (tuple(sorted(context)), int(v))


HINDSIGHT_CLASSES = {
    "countdown": CountdownHindsight,
    "jugs": JugsHindsight,
}
