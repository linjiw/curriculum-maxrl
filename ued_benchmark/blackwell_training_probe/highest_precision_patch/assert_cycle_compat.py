"""Isolated compatibility shim for the frozen parity runner's legacy helper API."""

from __future__ import annotations

from typing import Any


SHIM_VERSION = "legacy-assert-cycle-v1"


def install(grouped_runner_module: Any) -> dict[str, Any]:
    """Adapt the current helper API while retaining its optimizer-count assertion."""
    original_summary = grouped_runner_module._state_summary
    original_assert = grouped_runner_module._assert_cycle
    observations: list[int] = []

    def legacy_summary(state: Any) -> dict[str, Any]:
        summary = dict(original_summary(state))
        optimizer_steps = int(summary.pop("optimizer_step_applications"))
        gradient_updates = int(summary["n_grad_updates"])
        if optimizer_steps != gradient_updates:
            raise AssertionError(
                "one-epoch/one-minibatch optimizer application count drift: "
                f"optimizer={optimizer_steps} gradients={gradient_updates}"
            )
        observations.append(optimizer_steps)
        return summary

    def legacy_assert(summary: Any, *, cycle: int) -> None:
        augmented = dict(summary)
        expected_optimizer_steps = 0 if cycle == 1 else 1
        augmented["optimizer_step_applications"] = expected_optimizer_steps
        original_assert(
            augmented,
            cycle=cycle,
            expected_optimizer_step_applications=expected_optimizer_steps,
        )

    grouped_runner_module._state_summary = legacy_summary
    grouped_runner_module._assert_cycle = legacy_assert
    return {
        "shim_version": SHIM_VERSION,
        "optimizer_step_observations": observations,
        "legacy_summary": legacy_summary,
        "legacy_assert": legacy_assert,
    }
