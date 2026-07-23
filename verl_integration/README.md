# verl integration for the MaxRL repo

Files to integrate the curriculum teacher into the official MaxRL codebase
(https://github.com/tajwarfahim/maxrl, a verl fork):

- `curriculum.py` — copy to `verl/utils/curriculum.py`. FrontierTeacher
  (discounted Beta pseudo-counts + Thompson-style draws over the derived
  coefficient-mass utility)
  and CurriculumSampler (weighted, stateful sampler), plus a dataset wrapper
  that assigns contiguous positions after filtering/concatenation.
- `main_ppo.patch` — enables the sampler behind `+data.curriculum.enable=true`.
- `ray_trainer.patch` — observe hook after reward computation, wandb metrics,
  teacher state persisted/restored with checkpoints. Both normal global-step
  and alternate dataloader-only resume require the matching teacher state and
  fail loudly rather than combining a resumed sampler cursor with reset counts.
- `smollm_curriculum.sh` — SmolLM2-360M + GSM8K launch script (fill in paths).

Apply from the MaxRL repo root:

    cp curriculum.py <maxrl>/verl/utils/curriculum.py
    cd <maxrl> && git apply main_ppo.patch ray_trainer.patch

The integration deliberately does **not** use source `extra_info.index` as a
teacher-array slot: those IDs can become non-contiguous after filtering or
collide across concatenated files. `CurriculumIndexedDataset` injects a
separate `curriculum_index`, and malformed feedback fails loudly.

The coefficient-mass identity is literal for binary rewards and fixed group
size. The official MaxRL implementation uses an `N`-scaled,
epsilon-normalized equivalent, so the priority ordering is preserved at fixed
`N`; with thresholded continuous rewards it should be described as a proxy.
Weighted sampling changes the training-task distribution and is not
importance-corrected back to uniform.
