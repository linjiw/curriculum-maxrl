from __future__ import annotations

import contextlib
import hashlib
import io
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from curriculum_maxrl.group_law_flip import analyze_group_law_flip as analyzer


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _passk(value: float) -> dict[str, dict[str, float]]:
    return {str(level): {"1": value / 2.0, "8": value} for level in range(13)}


class CampaignFixture:
    def __init__(
        self,
        root: Path,
        seeds: tuple[int, ...],
        *,
        plugin_delta: float = 0.010,
        grouplaw_delta: float = 0.020,
        deliver: bool = True,
    ) -> None:
        self.root = root
        self.seeds = seeds
        self.deltas = {"plugin": plugin_delta, "grouplaw": grouplaw_delta}
        self.deliver = deliver
        root.mkdir()
        for seed in seeds:
            self.write_block(seed)

    def result_records(self, arm: str, seed: int, source: str, warm: str) -> list[dict]:
        teacher, family, exponent = {
            "plugin": ("group_law_plugin", "iid_plugin_from_count_law_mean", 32),
            "grouplaw": ("group_law_activity", "group_law_activity", None),
        }[arm]
        rows = [
            {
                "record_type": "config",
                "protocol": analyzer.PROTOCOL,
                "campaign": analyzer.CAMPAIGN_ID,
                "source_manifest": source,
                "arm": arm,
                "seed": seed,
                "teacher": teacher,
                "estimator": "maxrl",
                "score_estimator": "maxrl",
                "score_family": family,
                "effective_exponent": exponent,
                "rollouts": 32,
                "posterior_family": "count_law_moments",
                "posterior_sampling": "posterior_mean",
                "posterior_prior_p0": 0.5,
                "posterior_prior_mass": 0.0625,
                "posterior_decay": 0.7,
                "teacher_floor": 0.15,
                "seeds": {
                    "base": seed,
                    "sft": seed,
                    "rl": seed,
                    "teacher": seed + 77,
                    "eval_tasks": 202_608_130 + seed,
                    "eval_samples": 302_608_130 + seed,
                },
                "steps": 250,
                "tasks_per_step": 8,
                "lr": 1e-4,
                "d_model": 128,
                "n_layers": 6,
                "eval_every": 25,
                "eval_tasks_per_level": 32,
                "eval_samples": 8,
                "planned_rl_eval_count": 10,
                "sft_steps": 600,
                "teacher_power": 1.0,
                "hindsight": False,
                "hindsight_dense": False,
                "hindsight_to_teacher": False,
                "sft_checkpoint": f"/scratch/test/seed-{seed}-sft.pt",
                "sft_checkpoint_sha256": warm,
            },
            {
                "record_type": "evaluation",
                "protocol": analyzer.PROTOCOL,
                "phase": "post_sft",
                "completed_updates": 0,
                "passk": _passk(0.10),
            },
        ]
        for update in analyzer.EXPECTED_UPDATES:
            row = {
                "record_type": "evaluation",
                "protocol": analyzer.PROTOCOL,
                "phase": "rl",
                "completed_updates": update,
                "passk": _passk(0.10 + self.deltas[arm]),
            }
            if update == 250:
                row["final"] = True
            rows.append(row)
        return rows

    def telemetry_records(self, arm: str) -> list[dict]:
        rows = []
        optimizer_step = 0
        selected_level = 0 if arm == "plugin" or not self.deliver else 1
        for update in range(1, 251):
            optimizer_step += 1
            rows.append(
                {
                    "record_type": "telemetry",
                    "protocol": analyzer.PROTOCOL,
                    "completed_updates": update,
                    "selected_levels": [selected_level] * 8,
                    "group_k": [16] * 8,
                    "coefficient_mass": [1.0] * 8,
                    "coefficient_mass_total": 8.0,
                    "dead_groups": 0,
                    "optimizer_step_applied": True,
                    "optimizer_step": optimizer_step,
                }
            )
        return rows

    def write_block(self, seed: int) -> None:
        block = self.root / f"seed-{seed}"
        for name in ("results", "telemetry", "warmstarts", "checkpoints", "meta"):
            (block / name).mkdir(parents=True, exist_ok=True)
        source_copy = block / "meta" / "SOURCE_SHA256SUMS"
        source_copy.write_text("source fixture\n", encoding="utf-8")
        source_hash = _sha(source_copy)
        warmstart = block / "warmstarts" / f"seed-{seed}-sft.pt"
        warmstart.write_bytes(f"warm-{seed}".encode())
        warm_hash = _sha(warmstart)

        for arm in analyzer.ARMS:
            result = block / "results" / f"groupflip_{arm}_s{seed}.jsonl"
            result.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in self.result_records(arm, seed, source_hash, warm_hash)
                ),
                encoding="utf-8",
            )
            telemetry = block / "telemetry" / f"groupflip_{arm}_s{seed}.telemetry.jsonl"
            telemetry.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.telemetry_records(arm)),
                encoding="utf-8",
            )
            checkpoint = block / "checkpoints" / f"groupflip_{arm}_s{seed}.pt"
            checkpoint.write_bytes(f"checkpoint-{seed}-{arm}".encode())
            receipt = {
                "schema": "curriculum-maxrl/group-law-flip/arm-receipt/v1",
                "protocol": analyzer.PROTOCOL,
                "campaign": analyzer.CAMPAIGN_ID,
                "attempt": analyzer.ATTEMPT_ID,
                "arm": arm,
                "seed": seed,
                "completed_updates": 250,
                "source_manifest_sha256": source_hash,
                "sft_checkpoint_sha256": warm_hash,
                "paths": {
                    "result": f"results/{result.name}",
                    "telemetry": f"telemetry/{telemetry.name}",
                    "checkpoint": f"checkpoints/{checkpoint.name}",
                },
                "sha256": {
                    "result": _sha(result),
                    "telemetry": _sha(telemetry),
                    "checkpoint": _sha(checkpoint),
                },
            }
            (block / "meta" / f"{arm}.DONE.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        files = sorted(
            path for path in block.rglob("*")
            if path.is_file() and path.name not in {"SHA256SUMS", "COMPLETE"}
        )
        (block / "SHA256SUMS").write_text(
            "".join(f"{_sha(path)}  ./{path.relative_to(block).as_posix()}\n" for path in files),
            encoding="utf-8",
        )
        complete = {
            "schema": "curriculum-maxrl/group-law-flip/block-complete/v1",
            "protocol": analyzer.PROTOCOL,
            "campaign": analyzer.CAMPAIGN_ID,
            "attempt": analyzer.ATTEMPT_ID,
            "seed": seed,
            "completed_arms": list(analyzer.ARMS),
        }
        (block / "COMPLETE").write_text(
            json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


@pytest.fixture
def small_design(monkeypatch: pytest.MonkeyPatch) -> tuple[int, ...]:
    seeds = tuple(range(3001, 3009))
    monkeypatch.setattr(analyzer, "EXPECTED_SEEDS", seeds)
    return seeds


def test_complete_campaign_supported_and_deterministic(
    tmp_path: Path, small_design: tuple[int, ...]
) -> None:
    fixture = CampaignFixture(tmp_path / "campaign", small_design)
    validated = analyzer.validate_campaign(fixture.root)
    assert validated.mean_delivery_tv == pytest.approx(1.0)
    first = analyzer.analyze_validated(validated)
    second = analyzer.analyze_validated(validated)
    assert first["primary_grouplaw_minus_plugin"] == second["primary_grouplaw_minus_plugin"]
    primary = first["primary_grouplaw_minus_plugin"]
    assert primary["mean"] == pytest.approx(0.010)
    assert primary["sign_flip_p_two_sided_exact"] == pytest.approx(2 / 2**8)
    assert primary["decision"] == "supported"


def test_treatment_not_delivered_overrides_endpoint(
    tmp_path: Path, small_design: tuple[int, ...]
) -> None:
    fixture = CampaignFixture(tmp_path / "campaign", small_design, deliver=False)
    result = analyzer.analyze_campaign(fixture.root)
    assert result["treatment_delivery"]["passed"] is False
    assert result["primary_grouplaw_minus_plugin"]["decision"] == "treatment_not_delivered"


def test_rejects_partial_campaign_and_bad_telemetry(
    tmp_path: Path, small_design: tuple[int, ...]
) -> None:
    fixture = CampaignFixture(tmp_path / "campaign", small_design)
    missing = fixture.root / f"seed-{small_design[-1]}"
    missing.rename(tmp_path / "held-out-block")
    with pytest.raises(analyzer.AnalysisError, match="block set mismatch"):
        analyzer.validate_campaign(fixture.root)

    telemetry = tmp_path / "held-out-block" / "telemetry" / f"groupflip_plugin_s{small_design[-1]}.telemetry.jsonl"
    rows = telemetry.read_text(encoding="utf-8").splitlines()
    telemetry.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(analyzer.AnalysisError, match="exactly 250"):
        analyzer.load_telemetry(telemetry)


def test_hash_and_posterior_contracts_fail_closed(
    tmp_path: Path, small_design: tuple[int, ...]
) -> None:
    fixture = CampaignFixture(tmp_path / "campaign", small_design)
    block = fixture.root / f"seed-{small_design[0]}"
    result = block / "results" / f"groupflip_plugin_s{small_design[0]}.jsonl"
    result.write_text(result.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(analyzer.AnalysisError, match="hash mismatch"):
        analyzer.validate_campaign(fixture.root)

    clean = tmp_path / f"groupflip_plugin_s{small_design[0]}.jsonl"
    source = "a" * 64
    warm = "b" * 64
    rows = fixture.result_records("plugin", small_design[0], source, warm)
    rows[0]["posterior_sampling"] = "thompson"
    clean.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(analyzer.AnalysisError, match="posterior_sampling"):
        analyzer.load_run(clean)


def test_exact_sign_flip_matches_brute_force() -> None:
    differences = np.asarray([0.5, -0.25, 0.125, 0.375, -0.0625])
    observed = abs(float(differences.mean()))
    statistics = [
        abs(float(np.mean(differences * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=differences.size)
    ]
    expected = sum(value >= observed - 1e-15 for value in statistics) / len(statistics)
    assert analyzer.exact_sign_flip_p(differences) == expected


def test_cli_is_single_use_and_partial_failure_writes_nothing(
    tmp_path: Path, small_design: tuple[int, ...]
) -> None:
    fixture = CampaignFixture(tmp_path / "campaign", small_design)
    output = tmp_path / "analysis.json"
    assert analyzer.main([str(fixture.root), "--output", str(output)]) == 0
    assert output.is_file()
    second_output = tmp_path / "analysis-second.json"
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        assert analyzer.main([str(fixture.root), "--output", str(second_output)]) == 2
    assert not second_output.exists()
    assert "single-use" in stderr.getvalue()

    incomplete = CampaignFixture(tmp_path / "incomplete", small_design)
    (incomplete.root / f"seed-{small_design[-1]}").rename(tmp_path / "removed")
    forbidden = tmp_path / "forbidden.json"
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        assert analyzer.main([str(incomplete.root), "--output", str(forbidden)]) == 2
    assert not forbidden.exists()
    assert stdout.getvalue() == ""
    assert "cov_auc" not in stderr.getvalue()
