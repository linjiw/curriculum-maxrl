#!/usr/bin/env python3
"""Strict local Rung-3 E2E for both v4 arms and Phase-B closure."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
HELPERS = ROOT / "ued_benchmark" / "hopper_v4"
PROTOCOL = ROOT / "ued_benchmark" / "analysis" / "development_protocol_v2_tie_aware_draft.json"
CONFIGS = {
    "frontier": ROOT / "ued_benchmark" / "configs" / "maze_frontier_exact_grouped_n8_tie_aware_v4.json",
    "maxmc": ROOT / "ued_benchmark" / "configs" / "maze_maxmc_group_matched_4x8_b500_tie_aware_v4.json",
}
CONFIG_HASHES = {
    "frontier": "0e1b1907b319e42437d91ef4b19fef9ea39183a68e49069a17e337d7f78147f2",
    "maxmc": "a3cc3ddf387a3bb7cf3d9759c3f13e6a74f2c7e32de311ac344d54ef5e703ec6",
}
PROTOCOL_SHA = "1e4bd62be2412fa5291fde9d2c8750f30ed2e9c9f43afcda93d8ab552e4a3269"
CONTRACT_SHA = "3d5f3827a82a4f713314091289196a1c2909dd5d7c4c96dd532052c5706e832b"
APPLIED_SHA = "9b411f61ebc56bb93fc22cad6b19299c38eab2b696fa17f7783c7729e1db02ae"
COMMIT = "d053054c5290a04c1c4cd8b55704d999cad73e30"
TREE = "b0cace1fc54984e21a842f12d15d0b899e33d270"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run(command: Sequence[str], *, environment: Mapping[str, str], expect: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(command), cwd=ROOT, env=dict(environment), stdin=subprocess.DEVNULL,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600,
    )
    if completed.returncode != expect:
        raise AssertionError(
            f"command exit {completed.returncode}, expected {expect}: {command}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def launcher(python: Path, source: Path, script: Path, arguments: Sequence[str]) -> list[str]:
    code = (
        "import runpy,sys; source=sys.argv.pop(1); helper=sys.argv.pop(1); "
        "script=sys.argv.pop(1); sys.path[:0]=[source,helper]; sys.argv[0]=script; "
        "runpy.run_path(script,run_name='__main__')"
    )
    return [str(python), "-I", "-B", "-c", code, str(source / "src"), str(HELPERS), str(script), *arguments]


def archived_prerequisite(
    rung: str, result_dir: str, bundle_manifest_sha256: str
) -> dict[str, Any]:
    job_id = "11" if rung == "import" else "12"
    fields = [
        ("field", "value"),
        ("job_id", job_id),
        ("utc", "2026-08-14T12:00:00Z"),
        ("host", "local-fixture"),
        ("result_dir", result_dir),
        ("bundle_manifest_sha256", bundle_manifest_sha256),
        ("applied_overlay_manifest_sha256", APPLIED_SHA),
    ]
    if rung == "import":
        fields.append(("training_endpoint", "false"))
        complete_text = "complete\t2026-08-14T12:00:01Z\n"
    else:
        fields.extend((
            ("endpoint_class", "bounded_engineering_one_update"),
            ("actual_student_updates", "1"),
            ("paper_evidence", "false"),
            ("config_sha256", CONFIG_HASHES["frontier"]),
            ("input_closure_sha256", "7" * 64),
        ))
        complete_text = json.dumps(
            {
                "complete_schema": 2,
                "artifact_type": "frontier_exact_grouped_one_update_engineering",
                "job_id": job_id,
                "paper_evidence": False,
                "actual_ppo_updates": 1,
                "n_grad_updates": 1,
                "ppo_epochs": 5,
                "ppo_minibatches": 1,
                "optimizer_step_applications": 5,
                "resource_accounting_source": "python_resource_getrusage_self_and_monotonic_ns",
                "external_accounting_authority": "terminal_slurm_sacct",
                "terminal_sacct_included": False,
                "input_closure_sha256": "7" * 64,
                "sha256sums_sha256": "placeholder",
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    receipt_text = "".join(f"{key}\t{value}\n" for key, value in fields)
    receipt_sha = hashlib.sha256(receipt_text.encode("utf-8")).hexdigest()
    manifest_text = f"{receipt_sha}  receipt.tsv\n"
    manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    if rung == "one_update":
        complete = json.loads(complete_text)
        complete["sha256sums_sha256"] = manifest_sha
        complete_text = json.dumps(complete, indent=2, sort_keys=True) + "\n"

    def archived_file(text: str) -> dict[str, str]:
        return {
            "encoding": "utf-8",
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }

    files = {
        "receipt.tsv": archived_file(receipt_text),
        "SHA256SUMS": archived_file(manifest_text),
        "COMPLETE": archived_file(complete_text),
    }
    return {
        "result_dir": result_dir,
        "manifest_sha256": manifest_sha,
        "complete_sha256": files["COMPLETE"]["sha256"],
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "actual_student_updates": 0 if rung == "import" else 1,
        "paper_evidence": False,
        "analyzer_eligible": False,
        "archived_provenance": {"schema": 1, "files": files},
    }


def input_closure(arm: str, hashes: Mapping[str, str]) -> dict[str, Any]:
    fake_bundle = "b" * 64
    import_result = "/scratch/test/maxrl/tests/ued-minimax-v4-import/11"
    one_update_result = "/scratch/test/maxrl/tests/ued-minimax-v4-one-update/12"
    return {
        "schema": 1, "status": "frozen_before_phase_a",
        "purpose": "draft_engineering_development_only_no_endpoint_authorization_not_paper_evidence",
        "endpoint_class": "bounded_engineering_terminal_chain_components_v4",
        "paper_evidence": False, "analyzer_eligible": False,
        "endpoint_access_authorized": False, "production_authorized": False,
        "cost100_implemented": False, "max_student_updates": 1,
        "arm": arm, "training_seed": 101, "job_id": "local-test", "attempt": 1,
        "from_last_checkpoint": False, "archive_interval": 0,
        "periodic_checkpoint_used": False, "no_requeue": True,
        "source_commit": COMMIT, "source_tree": TREE,
        "bundle_manifest_sha256": fake_bundle,
        "overlay_manifest_sha256": "d" * 64,
        "applied_overlay_manifest_sha256": APPLIED_SHA,
        "environment_manifest_sha256": "e" * 64,
        "sbatch_sha256": hashes["sbatch"],
        "phase_a_driver_sha256": hashes["phase_a"],
        "training_driver_sha256": hashes["training"],
        "evaluation_driver_sha256": hashes["evaluation"],
        "assembler_sha256": hashes["assembler"],
        "finalizer_sha256": hashes["finalizer"],
        "protocol_sha256": PROTOCOL_SHA,
        "config_path": (
            "ued_benchmark/configs/maze_frontier_exact_grouped_n8_tie_aware_v4.json"
            if arm == "frontier"
            else "ued_benchmark/configs/maze_maxmc_group_matched_4x8_b500_tie_aware_v4.json"
        ),
        "config_sha256": CONFIG_HASHES[arm],
        "prerequisites": {
            "import": archived_prerequisite(
                "import", import_result, fake_bundle
            ),
            "one_update": archived_prerequisite(
                "one_update", one_update_result, fake_bundle
            ),
        },
    }


def exports(closure: Mapping[str, Any]) -> dict[str, str]:
    prereq = closure["prerequisites"]
    return {
        "UED_BUNDLE_DIR": "/scratch/test/maxrl/bundles/ued_minimax_v4_engineering/bbbbbbbbbbbbbbbbbbbb",
        "UED_BUNDLE_MANIFEST_SHA256": closure["bundle_manifest_sha256"],
        "UED_UPSTREAM_COMMIT": COMMIT, "UED_UPSTREAM_TREE": TREE,
        "UED_UPSTREAM_BUNDLE_SHA256": "a" * 64,
        "UED_OVERLAY_MANIFEST_SHA256": closure["overlay_manifest_sha256"],
        "UED_ENV_DIR": "/scratch/test/envs/ued-minimax-v2-aaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb",
        "UED_ENV_LOCK_SHA256": "5" * 64, "UED_ENV_FREEZE_SHA256": "6" * 64,
        "UED_ENV_MANIFEST_SHA256": closure["environment_manifest_sha256"],
        "UED_SBATCH_SHA256": closure["sbatch_sha256"],
        "UED_IMPORT_SMOKE_RESULT_DIR": prereq["import"]["result_dir"],
        "UED_IMPORT_SMOKE_MANIFEST_SHA256": prereq["import"]["manifest_sha256"],
        "UED_ONE_UPDATE_RESULT_DIR": prereq["one_update"]["result_dir"],
        "UED_ONE_UPDATE_MANIFEST_SHA256": prereq["one_update"]["manifest_sha256"],
        "UED_ARM": closure["arm"], "UED_CONFIG_SHA256": closure["config_sha256"],
        "UED_CONTRACT_SHA256": CONTRACT_SHA, "UED_PROTOCOL_SHA256": PROTOCOL_SHA,
        "UED_PHASE_A_DRIVER_SHA256": closure["phase_a_driver_sha256"],
        "UED_TRAINING_DRIVER_SHA256": closure["training_driver_sha256"],
        "UED_EVALUATION_DRIVER_SHA256": closure["evaluation_driver_sha256"],
        "UED_ASSEMBLER_SHA256": closure["assembler_sha256"],
        "UED_FINALIZER_SHA256": closure["finalizer_sha256"],
    }


def tree_digest(root: Path) -> str:
    lines = [
        f"{digest(path)}  ./{path.relative_to(root).as_posix()}\n"
        for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix())
    ]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def write_phase_b_receipts(root: Path, components: Path, closure: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    exported = exports(closure)
    export_arg = "--export=" + ",".join(
        f"{key}={exported[key]}" for key in sorted(exported)
    )
    remote_script = "/tmp/ued_minimax_v4_terminal_chain_smoke.sbatch"
    submit_line = shlex_join(["sbatch", "--parsable", export_arg, remote_script])
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = now - timedelta(seconds=1)
    submit = start - timedelta(seconds=1)
    submission = root / "submission.tsv"
    header = (
        "job_id\tutc\thost\tlocal_script\tlocal_sha256\tremote_script\t"
        "remote_sha256\toutput_path\tremote_receipt\tsbatch_args"
    )
    submission.write_text(
        header + "\n" + "\t".join((
            "local-test", (submit + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), "localhost",
            str(ROOT / "hopper" / "sbatch" / "ued_minimax_v4_terminal_chain_smoke.sbatch"),
            closure["sbatch_sha256"], remote_script, closure["sbatch_sha256"],
            "/tmp/ued-v4-terminal.out", "/tmp/job-local-test.tsv", export_arg,
        )) + "\n", encoding="utf-8",
    )
    terminal = root / "terminal.tsv"
    terminal_row = "|".join((
        "local-test", "ued-v4-terminal", "local", "COMPLETED", "0:0", "1", "2", "15G",
        "localhost", submit.strftime("%Y-%m-%dT%H:%M:%S"),
        start.strftime("%Y-%m-%dT%H:%M:%S"), now.strftime("%Y-%m-%dT%H:%M:%S"),
        "cpu=2,gres/gpu=1", "local", "45", "0", str(ROOT),
        "/tmp/ued-v4-terminal.out", "/tmp/ued-v4-terminal.out", submit_line,
    ))
    terminal.write_text(
        "terminal_receipt_schema\t2\n"
        f"retrieved_utc\t{(now + timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        f"retrieved_epoch\t{int((now + timedelta(seconds=1)).timestamp())}\n"
        f"terminal_end_epoch\t{int(now.timestamp())}\n"
        f"terminal_header\tJobIDRaw|JobName|Partition|State|ExitCode|ElapsedRaw|AllocCPUS|ReqMem|NodeList|Submit|Start|End|AllocTRES|QOS|TimelimitRaw|Restarts|WorkDir|StdOut|StdErr|SubmitLine\n"
        f"terminal_row\t{terminal_row}\n"
        "resource_header\tJobIDRaw|MaxRSS|TRESUsageInMax\n"
        "resource_row\tlocal-test|1M|gres/gpumem=1M\n",
        encoding="utf-8",
    )
    fetched = root / "fetch.tsv"
    fetched_digest = tree_digest(components)
    fetched.write_text(
        "fetch_receipt_schema\t2\n"
        f"fetch_started_utc\t{(now + timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        f"fetch_started_epoch\t{int((now + timedelta(seconds=1)).timestamp())}\n"
        f"retrieved_utc\t{(now + timedelta(seconds=2)).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        f"retrieved_epoch\t{int((now + timedelta(seconds=2)).timestamp())}\n"
        f"terminal_end_epoch\t{int(now.timestamp())}\n"
        f"terminal_receipt_sha256\t{digest(terminal)}\n"
        f"remote_path\t/tmp/{closure['arm']}-components\n"
        "remote_type\tdir\n"
        f"remote_digest\t{fetched_digest}\n"
        "manifest_verified\t1\n"
        f"local_path\t{components}\n"
        f"local_digest\t{fetched_digest}\n",
        encoding="utf-8",
    )
    return terminal, submission, fetched


def shlex_join(parts: Sequence[str]) -> str:
    import shlex
    return shlex.join(parts)


def rebuild_sidecar(sidecar: Path, run_id: str, arm: str) -> None:
    payloads = ["plr-replay-snapshot.json", "training-receipt.json"]
    (sidecar / "SHA256SUMS").write_text(
        "".join(f"{digest(sidecar / name)}  {name}\n" for name in payloads), encoding="utf-8"
    )
    write_json(sidecar / "COMPLETE", {
        "schema": 1, "status": "complete", "run_id": run_id, "arm": arm,
        "sha256sums_sha256": digest(sidecar / "SHA256SUMS"), "file_count": 2,
    })


def rebuild_package_root(package: Path) -> None:
    payloads = sorted(
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file()
        and path.relative_to(package).as_posix() not in {"SHA256SUMS", "COMPLETE"}
    )
    (package / "SHA256SUMS").write_text(
        "".join(f"{digest(package / name)}  {name}\n" for name in payloads),
        encoding="utf-8",
    )
    complete = json.loads((package / "COMPLETE").read_text(encoding="utf-8"))
    complete["sha256sums_sha256"] = digest(package / "SHA256SUMS")
    complete["file_count"] = len(payloads)
    write_json(package / "COMPLETE", complete)


def main() -> int:
    source = Path(os.environ.get("MINIMAX_SOURCE_DIR", "/tmp/root-minimax-260814")).resolve()
    python = Path(os.environ.get("UED_CPU_PYTHON", "/data/robotixx/ued_bench/envs/minimax-jax0431-cpu/bin/python"))
    if not python.is_file() or not (source / ".git").exists():
        raise AssertionError("pinned CPU Python and minimax source are required")
    with tempfile.TemporaryDirectory(prefix="ued-v4-r3-local.") as raw:
        temporary = Path(raw)
        patched = temporary / "minimax"
        subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(source), str(patched)], check=True)
        subprocess.run(["git", "-C", str(patched), "checkout", "--quiet", "--detach", COMMIT], check=True)
        subprocess.run([
            str(python), "-I", "-B", str(ROOT / "ued_benchmark" / "scripts" / "apply_minimax_overlay_v4.py"),
            "--target", str(patched), "--apply",
        ], check=True, stdout=subprocess.DEVNULL)
        assert digest(patched / ".frontierrl_overlay.json") == APPLIED_SHA
        files = {
            "phase_a": HELPERS / "run_terminal_phase_a_v4.py",
            "training": HELPERS / "run_matched_terminal_v4.py",
            "evaluation": HELPERS / "evaluate_matched_terminal_v4.py",
            "assembler": HELPERS / "assemble_matched_run_v4.py",
            "finalizer": ROOT / "hopper" / "finalize_ued_minimax_v4_terminal_chain.py",
            "sbatch": ROOT / "hopper" / "sbatch" / "ued_minimax_v4_terminal_chain_smoke.sbatch",
        }
        hashes = {key: digest(path) for key, path in files.items()}
        environment = os.environ.copy()
        environment.update({
            "JAX_PLATFORMS": "cpu", "JAX_PLATFORM_NAME": "cpu", "PYTHONPATH": "",
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "WANDB_MODE": "disabled",
            "TMPDIR": str(temporary), "XDG_CACHE_HOME": str(temporary / "cache"),
            "JAX_COMPILATION_CACHE_DIR": str(temporary / "jax-cache"),
        })
        (temporary / "cache").mkdir(); (temporary / "jax-cache").mkdir()
        finite_gate = r'''
import hashlib,json,sys
from pathlib import Path
sys.path[:0]=[sys.argv[1],sys.argv[2]]
import run_terminal_phase_a_v4 as phase
source=Path(sys.argv[3]); output=Path(sys.argv[4])
base=json.loads(source.read_text(encoding="utf-8"))
for index,value in enumerate((0.0,float("nan"),float("inf"))):
    document=json.loads(json.dumps(base)); document["args"]["plr_temp"]=[value]
    path=output/f"bad-temp-{index}.json"
    path.write_text(json.dumps(document)+"\n",encoding="utf-8")
    phase.HASHES["frontier_config"]=hashlib.sha256(path.read_bytes()).hexdigest()
    try: phase._validate_config(path,"frontier")
    except phase.PhaseAError as exc: assert "finite and positive" in str(exc)
    else: raise AssertionError(f"accepted bad temperature {value!r}")
'''
        run(
            [str(python), "-I", "-B", "-c", finite_gate, str(patched / "src"),
             str(HELPERS), str(CONFIGS["frontier"]), str(temporary)],
            environment=environment,
        )
        cells: dict[str, Path] = {}
        campaign_hashes: set[str] = set()
        for arm in ("frontier", "maxmc"):
            arm_root = temporary / arm
            arm_root.mkdir()
            closure = input_closure(arm, hashes)
            closure_path = arm_root / "INPUT_CLOSURE.json"
            write_json(closure_path, closure)
            if arm == "frontier":
                sys.path[:0] = [str(patched / "src"), str(HELPERS)]
                import run_terminal_phase_a_v4 as phase_a

                validation_cli = SimpleNamespace(
                    arm=arm, job_id="local-test",
                    bundle_manifest_sha256=closure["bundle_manifest_sha256"],
                    overlay_manifest_sha256=closure["overlay_manifest_sha256"],
                    applied_overlay_manifest_sha256=closure["applied_overlay_manifest_sha256"],
                    environment_manifest_sha256=closure["environment_manifest_sha256"],
                    sbatch_sha256=closure["sbatch_sha256"],
                    expected_phase_a_driver_sha256=closure["phase_a_driver_sha256"],
                    expected_training_driver_sha256=closure["training_driver_sha256"],
                    expected_evaluation_driver_sha256=closure["evaluation_driver_sha256"],
                    expected_assembler_sha256=closure["assembler_sha256"],
                    expected_finalizer_sha256=closure["finalizer_sha256"],
                )
                phase_a._validate_input_closure(
                    closure, validation_cli, CONFIG_HASHES[arm]
                )
                mutations = (
                    ("paper_evidence", True),
                    ("production_authorized", True),
                    ("cost100_implemented", True),
                    ("from_last_checkpoint", True),
                    ("periodic_checkpoint_used", True),
                    ("no_requeue", False),
                    ("attempt", 2),
                    ("max_student_updates", 100),
                    ("training_seed", 102),
                    ("job_id", "wrong-job"),
                )
                for key, value in mutations:
                    changed = copy.deepcopy(closure); changed[key] = value
                    try:
                        phase_a._validate_input_closure(
                            changed, validation_cli, CONFIG_HASHES[arm]
                        )
                    except phase_a.PhaseAError:
                        pass
                    else:
                        raise AssertionError(f"input closure accepted {key}={value!r}")
                changed = copy.deepcopy(closure)
                changed["prerequisites"]["one_update"]["bundle_manifest_sha256"] = "9" * 64
                try:
                    phase_a._validate_input_closure(
                        changed, validation_cli, CONFIG_HASHES[arm]
                    )
                except phase_a.PhaseAError as exc:
                    assert "cross-bundle" in str(exc)
                else:
                    raise AssertionError("input closure accepted cross-bundle R2")
                changed = copy.deepcopy(closure)
                archived_receipt = changed["prerequisites"]["import"][
                    "archived_provenance"
                ]["files"]["receipt.tsv"]
                archived_receipt["text"] += "injected\tvalue\n"
                archived_receipt["sha256"] = hashlib.sha256(
                    archived_receipt["text"].encode("utf-8")
                ).hexdigest()
                try:
                    phase_a._validate_input_closure(
                        changed, validation_cli, CONFIG_HASHES[arm]
                    )
                except phase_a.PhaseAError as exc:
                    assert "receipt/manifest binding" in str(exc)
                else:
                    raise AssertionError(
                        "input closure accepted forged archived prerequisite receipt"
                    )
                changed = copy.deepcopy(closure)
                archived_complete = changed["prerequisites"]["one_update"][
                    "archived_provenance"
                ]["files"]["COMPLETE"]
                complete_record = json.loads(archived_complete["text"])
                complete_record["input_closure_sha256"] = "9" * 64
                archived_complete["text"] = json.dumps(
                    complete_record, indent=2, sort_keys=True
                ) + "\n"
                archived_complete["sha256"] = hashlib.sha256(
                    archived_complete["text"].encode("utf-8")
                ).hexdigest()
                changed["prerequisites"]["one_update"]["complete_sha256"] = (
                    archived_complete["sha256"]
                )
                try:
                    phase_a._validate_input_closure(
                        changed, validation_cli, CONFIG_HASHES[arm]
                    )
                except phase_a.PhaseAError as exc:
                    assert "completion semantics" in str(exc)
                else:
                    raise AssertionError(
                        "input closure accepted mixed one-update receipt/completion"
                    )
            components = arm_root / "components"
            args = [
                "--arm", arm, "--job-id", "local-test", "--protocol", str(PROTOCOL),
                "--config", str(CONFIGS[arm]), "--patched-source-dir", str(patched),
                "--git-executable", shutil.which("git") or "/usr/bin/git", "--python", str(python),
                "--input-closure", str(closure_path),
                "--expected-input-closure-sha256", digest(closure_path),
                "--bundle-manifest-sha256", closure["bundle_manifest_sha256"],
                "--overlay-manifest-sha256", closure["overlay_manifest_sha256"],
                "--applied-overlay-manifest-sha256", APPLIED_SHA,
                "--environment-manifest-sha256", closure["environment_manifest_sha256"],
                "--sbatch-sha256", hashes["sbatch"],
                "--expected-phase-a-driver-sha256", hashes["phase_a"],
                "--expected-training-driver-sha256", hashes["training"],
                "--expected-evaluation-driver-sha256", hashes["evaluation"],
                "--expected-assembler-sha256", hashes["assembler"],
                "--expected-finalizer-sha256", hashes["finalizer"],
                "--output-dir", str(components), "--local-test-mode",
            ]
            completed = run(launcher(python, patched, files["phase_a"], args), environment=environment)
            assert "V4_TERMINAL_PHASE_A_COMPLETE" in completed.stdout
            complete = json.loads((components / "COMPONENTS_COMPLETE.json").read_text(encoding="utf-8"))
            assert complete["paper_evidence"] is False and complete["analyzer_eligible"] is False
            assert complete["actual_student_updates"] == 1 and complete["actual_external_evaluation"] is True
            assert complete["raw_evaluation_records"] == 30 and complete["phase_b_required"] is True
            campaign = json.loads((components / "campaign-manifest.json").read_text(encoding="utf-8"))
            assert len(campaign["submissions"]) == 1 and campaign["submissions"][0]["arm"] == arm
            campaign_hashes.add(digest(components / "campaign-manifest.json"))
            evaluation = json.loads(
                (
                    components
                    / "evaluation-package"
                    / "evaluation-integrity-receipt.json"
                ).read_text(encoding="utf-8")
            )
            assert evaluation["performance_fields_included"] is False
            assert evaluation["evaluation_receipt_sha256"] == digest(
                components / "evaluation-package" / "evaluation-receipt.json"
            )
            assert evaluation["synthetic_test_mode"] is False
            assert evaluation["raw_results"]["record_count"] == 30
            accounting = evaluation["evaluation_transition_accounting"]
            assert accounting["budgeted_primary_max_transitions"] == 13_500
            assert accounting["effective_primary_transitions"] == 13_500
            snapshot = json.loads((components / "training-sidecar" / "plr-replay-snapshot.json").read_text(encoding="utf-8"))
            assert snapshot["arm"] == arm and snapshot["kind"] == "tie_aware_plr_buffer_safe_snapshot"
            assert snapshot["replay_distribution"]["tie_aware_score_ranks"] is True
            assert snapshot["sampling_diagnostics"]["replay_group_draw_count"] == 4
            if arm == "frontier":
                assert snapshot["stored_score_validation"] is not None
                assert "trial_count" in snapshot["slots"][0]
            else:
                assert snapshot["stored_score_validation"] is None
                assert "trial_count" not in snapshot["slots"][0]
            terminal, submission, fetched = write_phase_b_receipts(arm_root, components, closure)
            package = arm_root / "package"
            final_args = [
                "--components-dir", str(components), "--protocol", str(PROTOCOL),
                "--assembler", str(files["assembler"]), "--python", str(python),
                "--terminal-receipt", str(terminal), "--submission-receipt", str(submission),
                "--fetch-receipt", str(fetched), "--job-id", "local-test", "--arm", arm,
                "--expected-components-manifest-sha256", digest(components / "SHA256SUMS"),
                "--expected-assembler-sha256", hashes["assembler"],
                "--expected-finalizer-sha256", hashes["finalizer"],
                "--expected-sbatch-sha256", hashes["sbatch"], "--output-dir", str(package),
                "--local-test-mode",
            ]
            final = run([str(python), "-I", "-B", str(files["finalizer"]), *final_args], environment=environment)
            assert "V4_TERMINAL_FINALIZATION_COMPLETE" in final.stdout
            package_complete = json.loads((package / "COMPLETE").read_text(encoding="utf-8"))
            assert package_complete["paper_evidence"] is False and package_complete["analyzer_eligible"] is False
            manifest = json.loads((package / "run-manifest.json").read_text(encoding="utf-8"))
            assert manifest["production_analyzer_invoked"] is False
            assert manifest["performance_values_inspected"] is False
            assert digest(package / "training-plr-replay-snapshot.json") == digest(package / "training-sidecar" / "plr-replay-snapshot.json")
            scheduler = json.loads((package / "scheduler.json").read_text(encoding="utf-8"))
            assert scheduler["max_rss_bytes"] == 1024 * 1024
            assert scheduler["resource_rows"] == [{
                "job_id": "local-test",
                "max_rss": "1M",
                "tres_usage_in_max": "gres/gpumem=1M",
            }]
            receipt_archive = package / "phase-b-receipts"
            assert {path.name for path in receipt_archive.iterdir()} == {
                "terminal.tsv", "submission.tsv", "fetch.tsv", "SHA256SUMS",
                "COMPLETE",
            }
            assert manifest["phase_b_receipts_manifest_sha256"] == digest(
                receipt_archive / "SHA256SUMS"
            )
            if arm == "frontier":
                complete_path = package / "COMPLETE"
                complete_bytes = complete_path.read_bytes()
                bad_complete = json.loads(complete_bytes)
                bad_complete["run_id"] = "injected-run"
                write_json(complete_path, bad_complete)
                refused = run(
                    launcher(
                        python, patched, files["assembler"],
                        ["--validate-only", str(package)],
                    ),
                    environment=environment, expect=1,
                )
                assert "completion identity drift" in refused.stderr
                complete_path.write_bytes(complete_bytes)

                flat_snapshot = package / "training-plr-replay-snapshot.json"
                flat_snapshot_bytes = flat_snapshot.read_bytes()
                injected = json.loads(flat_snapshot_bytes)
                injected["run_id"] = "injected-run"
                write_json(flat_snapshot, injected)
                rebuild_package_root(package)
                refused = run(
                    launcher(
                        python, patched, files["assembler"],
                        ["--validate-only", str(package)],
                    ),
                    environment=environment, expect=1,
                )
                assert "flat/source PLR snapshot drift" in refused.stderr
                flat_snapshot.write_bytes(flat_snapshot_bytes)
                rebuild_package_root(package)

                archived_terminal = receipt_archive / "terminal.tsv"
                archived_terminal_bytes = archived_terminal.read_bytes()
                archived_terminal.write_bytes(archived_terminal_bytes + b"injected\tvalue\n")
                rebuild_package_root(package)
                refused = run(
                    launcher(
                        python, patched, files["assembler"],
                        ["--validate-only", str(package)],
                    ),
                    environment=environment, expect=1,
                )
                assert "payload hash drift: terminal.tsv" in refused.stderr
                archived_terminal.write_bytes(archived_terminal_bytes)
                rebuild_package_root(package)

                wrong_python_args = list(final_args)
                wrong_python_args[wrong_python_args.index("--python") + 1] = "/bin/true"
                wrong_python_args[wrong_python_args.index(str(package))] = str(
                    arm_root / "wrong-python-package"
                )
                refused = run(
                    [
                        str(python), "-I", "-B", str(files["finalizer"]),
                        *wrong_python_args,
                    ],
                    environment=environment, expect=1,
                )
                assert "Python executable identity drift" in refused.stderr

                # Caller paths are validated before resolution, so symlinked
                # components and outputs cannot be laundered into Phase B.
                components_link = arm_root / "components-link"
                components_link.symlink_to(components, target_is_directory=True)
                linked_args = list(final_args)
                linked_args[linked_args.index(str(components))] = str(components_link)
                linked_args[linked_args.index(str(package))] = str(arm_root / "linked-components-package")
                refused = run(
                    [str(python), "-I", "-B", str(files["finalizer"]), *linked_args],
                    environment=environment, expect=1,
                )
                assert "noncanonical or contains a symlink" in refused.stderr
                components_link.unlink()

                dangling_output = arm_root / "dangling-package"
                dangling_output.symlink_to(arm_root / "missing-target", target_is_directory=True)
                dangling_args = list(final_args)
                dangling_args[dangling_args.index(str(package))] = str(dangling_output)
                refused = run(
                    [str(python), "-I", "-B", str(files["finalizer"]), *dangling_args],
                    environment=environment, expect=1,
                )
                assert "output exists" in refused.stderr
                dangling_output.unlink()

                overlap_args = list(final_args)
                overlap_args[overlap_args.index(str(package))] = str(components / "phase-b-output")
                refused = run(
                    [str(python), "-I", "-B", str(files["finalizer"]), *overlap_args],
                    environment=environment, expect=1,
                )
                assert "output/components overlap" in refused.stderr

                dotdot_output = Path(
                    f"{arm_root}/unused-parent/../components/phase-b-dotdot"
                )
                dotdot_args = list(final_args)
                dotdot_args[dotdot_args.index(str(package))] = str(dotdot_output)
                refused = run(
                    [str(python), "-I", "-B", str(files["finalizer"]), *dotdot_args],
                    environment=environment, expect=1,
                )
                assert "output must be canonical absolute" in refused.stderr

                alias_parent = arm_root / "output-parent-alias"
                alias_parent.symlink_to(arm_root, target_is_directory=True)
                alias_args = list(final_args)
                alias_args[alias_args.index(str(package))] = str(
                    alias_parent / "aliased-package"
                )
                refused = run(
                    [str(python), "-I", "-B", str(files["finalizer"]), *alias_args],
                    environment=environment, expect=1,
                )
                assert "output parent is noncanonical or contains a symlink" in refused.stderr
                alias_parent.unlink()

                wrong_arm_args = list(final_args)
                wrong_arm_args[wrong_arm_args.index("--arm") + 1] = "maxmc"
                wrong_arm_args[wrong_arm_args.index(str(package))] = str(arm_root / "wrong-arm-package")
                refused = run(
                    [str(python), "-I", "-B", str(files["finalizer"]), *wrong_arm_args],
                    environment=environment, expect=1,
                )
                assert "component arm/job drift" in refused.stderr

                bad_restart = arm_root / "restarted-terminal.tsv"
                bad_restart.write_text(
                    terminal.read_text(encoding="utf-8").replace("|45|0|", "|45|1|"),
                    encoding="utf-8",
                )
                restart_args = list(final_args)
                restart_args[restart_args.index(str(terminal))] = str(bad_restart)
                restart_args[restart_args.index(str(package))] = str(arm_root / "restarted-package")
                refused = run(
                    [str(python), "-I", "-B", str(files["finalizer"]), *restart_args],
                    environment=environment, expect=1,
                )
                assert "time/restart drift" in refused.stderr

                bad_name = arm_root / "wrong-name-terminal.tsv"
                bad_name.write_text(
                    terminal.read_text(encoding="utf-8").replace(
                        "|ued-v4-terminal|", "|ued-v3-terminal|"
                    ),
                    encoding="utf-8",
                )
                name_args = list(final_args)
                name_args[name_args.index(str(terminal))] = str(bad_name)
                name_args[name_args.index(str(package))] = str(arm_root / "wrong-name-package")
                refused = run(
                    [str(python), "-I", "-B", str(files["finalizer"]), *name_args],
                    environment=environment, expect=1,
                )
                assert "job name drift" in refused.stderr

                fetch_lines = fetched.read_text(encoding="utf-8").splitlines()
                terminal_end = int(next(
                    line.split("\t", 1)[1]
                    for line in fetch_lines if line.startswith("terminal_end_epoch\t")
                ))
                before = datetime.fromtimestamp(terminal_end - 1, timezone.utc)
                preterminal_fetch = arm_root / "preterminal-fetch.tsv"
                preterminal_fetch.write_text(
                    "\n".join(
                        f"fetch_started_utc\t{before.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                        if line.startswith("fetch_started_utc\t")
                        else f"fetch_started_epoch\t{terminal_end - 1}"
                        if line.startswith("fetch_started_epoch\t")
                        else line
                        for line in fetch_lines
                    ) + "\n",
                    encoding="utf-8",
                )
                fetch_args = list(final_args)
                fetch_args[fetch_args.index(str(fetched))] = str(preterminal_fetch)
                fetch_args[fetch_args.index(str(package))] = str(arm_root / "preterminal-package")
                refused = run(
                    [str(python), "-I", "-B", str(files["finalizer"]), *fetch_args],
                    environment=environment, expect=1,
                )
                assert "fetch occurred before terminal receipt" in refused.stderr
            # A preterminal/failed receipt is rejected before any output appears.
            failed_terminal = arm_root / "failed-terminal.tsv"
            failed_terminal.write_text(terminal.read_text(encoding="utf-8").replace("|COMPLETED|0:0|", "|FAILED|1:0|"), encoding="utf-8")
            failed_args = list(final_args)
            failed_args[failed_args.index(str(terminal))] = str(failed_terminal)
            failed_args[failed_args.index(str(package))] = str(arm_root / "failed-package")
            refused = run([str(python), "-I", "-B", str(files["finalizer"]), *failed_args], environment=environment, expect=1)
            assert "FINALIZATION_REFUSED" in refused.stderr
            cells[arm] = components
        assert len(campaign_hashes) == 2, "the two arms reused one campaign"

        # Rebuild a valid inner sidecar closure around a cross-arm snapshot;
        # semantic validation must still reject it rather than relying on hashes.
        negative = temporary / "cross-arm"
        (negative / "training-sidecar").parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(cells["frontier"] / "training-sidecar", negative / "training-sidecar")
        sidecar = negative / "training-sidecar"
        snapshot_path = sidecar / "plr-replay-snapshot.json"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8")); snapshot["arm"] = "maxmc"
        write_json(snapshot_path, snapshot)
        receipt_path = sidecar / "training-receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["plr_snapshot"]["sha256"] = digest(snapshot_path); write_json(receipt_path, receipt)
        rebuild_sidecar(sidecar, "engineering-frontier-s101", "frontier")
        code = (
            "import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
            "import assemble_matched_run_v4 as a; "
            "a._validate_replay_sidecar(Path(sys.argv[2]),"
            "{'run_id':'engineering-frontier-s101','arm':'frontier','training_seed':101})"
        )
        refused = run([str(python), "-I", "-B", "-c", code, str(HELPERS), str(negative)], environment=environment, expect=1)
        assert "snapshot run/arm/seed drift" in refused.stderr
    print("UED_MINIMAX_V4_R3_LOCAL_PASS arms=2 actual_eval_records=60 sealed_values=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
