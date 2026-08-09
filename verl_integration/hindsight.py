"""Release-safe hindsight integration.

``vendored/hindsight.py`` is the immutable execution snapshot used by the
reported experiments.  This module preserves that implementation while
overriding only its unsafe response-number substitution.  See README.md for
the three-file deployment route into the MaxRL/verl fork.
"""

from __future__ import annotations

if __package__ == "verl.utils":  # copied into the MaxRL execution fork
    from .hindsight_snapshot import (  # type: ignore[import-not-found]
        CountdownHindsight as _SnapshotCountdownHindsight,
        JugsHindsight as _SnapshotJugsHindsight,
    )
    from .response_goal_rewrite import rewrite_response_goal
else:  # imported and tested from this paper repository
    from .response_goal_rewrite import rewrite_response_goal
    from .vendored.hindsight import (
        CountdownHindsight as _SnapshotCountdownHindsight,
        JugsHindsight as _SnapshotJugsHindsight,
    )


class CountdownHindsight(_SnapshotCountdownHindsight):
    """Countdown relabeler with conservative goal-context rewriting."""

    def _rewrite_response_text(self, resp_text: str, old_target: int,
                               v: int) -> str:
        return rewrite_response_goal(resp_text, old_target, v,
                                     domain="countdown")


class JugsHindsight(_SnapshotJugsHindsight):
    """Jugs relabeler with conservative goal-context rewriting."""

    def _rewrite_response_text(self, resp_text: str, old_target: int,
                               v: int) -> str:
        return rewrite_response_goal(resp_text, old_target, v, domain="jugs")


HINDSIGHT_CLASSES = {
    "countdown": CountdownHindsight,
    "jugs": JugsHindsight,
}
