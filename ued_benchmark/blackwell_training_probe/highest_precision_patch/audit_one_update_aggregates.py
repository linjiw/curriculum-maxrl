#!/usr/bin/env python3
"""Exhaustively audit a bounded one-update receipt against the frozen reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "HIGHEST_PRECISION_ONE_UPDATE_PROTOCOL.json"
PROTOCOL_SHA256 = "ba0b6fd30de472554d732308017cb8d3c28f7ddef0549631fc5fe907610ec4c3"
REFERENCE_RECEIPT_SHA256 = "1005e3c907c38061f23c46ef8b8b24016818603d4bf42bfd1555afe073b3c8e9"


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _exact_gate(label: str, reference: Any, candidate: Any) -> dict[str, Any]:
    return {
        "label": label,
        "reference": reference,
        "candidate": candidate,
        "pass": reference == candidate,
    }


def _numeric_gate(
    label: str,
    reference: float,
    candidate: float,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    left = float(reference)
    right = float(candidate)
    absolute = abs(left - right)
    threshold = atol + rtol * abs(right)
    relative = absolute / max(abs(left), atol)
    return {
        "label": label,
        "reference": left,
        "candidate": right,
        "absolute_error": absolute,
        "relative_error": relative,
        "allowed_absolute_error": threshold,
        "rtol": rtol,
        "atol": atol,
        "pass": math.isfinite(left) and math.isfinite(right) and absolute <= threshold,
    }


def _leaf_gates(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    stage: str,
    rtol: float,
    atol: float,
    require_exact_hash: bool,
) -> dict[str, Any]:
    exact: list[dict[str, Any]] = []
    hashes: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    sentinels: list[dict[str, Any]] = []
    exact.append(
        _exact_gate(
            f"{stage}.leaf_count", reference.get("leaf_count"), candidate.get("leaf_count")
        )
    )
    exact.append(
        _exact_gate(
            f"{stage}.structure_sha256",
            reference.get("structure_sha256"),
            candidate.get("structure_sha256"),
        )
    )
    candidate_by_path = {
        item.get("path"): item for item in candidate.get("leaves", [])
    }
    ordered_candidate_paths = [item.get("path") for item in candidate.get("leaves", [])]
    ordered_reference_paths = [item.get("path") for item in reference.get("leaves", [])]
    exact.append(
        _exact_gate(f"{stage}.ordered_paths", ordered_reference_paths, ordered_candidate_paths)
    )
    for before in reference.get("leaves", []):
        path = before["path"]
        after = candidate_by_path.get(path)
        exact.append(_exact_gate(f"{stage}.{path}.present", True, after is not None))
        if after is None:
            for metric in ("abs_sum", "squared_l2", "max_abs"):
                aggregates.append(
                    {
                        "label": f"{stage}.{path}.{metric}",
                        "reference": before.get(metric),
                        "candidate": None,
                        "rtol": rtol,
                        "atol": atol,
                        "pass": False,
                        "reason": "candidate leaf missing",
                    }
                )
            continue
        for field in ("path", "shape", "dtype", "size"):
            exact.append(
                _exact_gate(
                    f"{stage}.{path}.{field}", before.get(field), after.get(field)
                )
            )
        hash_gate = _exact_gate(
            f"{stage}.{path}.sha256", before.get("sha256"), after.get("sha256")
        )
        hash_gate["required"] = require_exact_hash
        hashes.append(hash_gate)
        for metric in (
            "nan_count",
            "infinite_count",
            "positive_inf_count",
            "negative_inf_count",
        ):
            if metric in before or metric in after:
                sentinels.append(
                    _exact_gate(
                        f"{stage}.{path}.{metric}", before.get(metric), after.get(metric)
                    )
                )
        for metric in ("abs_sum", "squared_l2", "max_abs"):
            if metric not in before or metric not in after:
                aggregates.append(
                    {
                        "label": f"{stage}.{path}.{metric}",
                        "reference": before.get(metric),
                        "candidate": after.get(metric),
                        "rtol": rtol,
                        "atol": atol,
                        "pass": False,
                        "reason": "numeric aggregate missing",
                    }
                )
            else:
                aggregates.append(
                    _numeric_gate(
                        f"{stage}.{path}.{metric}",
                        before[metric],
                        after[metric],
                        rtol=rtol,
                        atol=atol,
                    )
                )
    return {
        "exact": exact,
        "hashes": hashes,
        "sentinels": sentinels,
        "aggregates": aggregates,
    }


def compare_documents(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    backend: str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    require(backend in ("cpu", "gpu"), "unsupported backend")
    tolerances = protocol["tolerances"][backend]
    stat_rtol = float(tolerances["stat_rtol"])
    stat_atol = float(tolerances["stat_atol"])
    leaf_rtol = float(tolerances["leaf_aggregate_rtol"])
    leaf_atol = float(tolerances["leaf_aggregate_atol"])

    exact: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for label, expected, actual in (
        ("candidate.backend", backend, candidate.get("backend")),
        ("candidate.lane", "modern", candidate.get("lane")),
        ("candidate.paper_evidence", False, candidate.get("paper_evidence")),
        ("candidate.ood_evaluation", False, candidate.get("ood_evaluation")),
        ("candidate.max_student_updates", 1, candidate.get("max_student_updates")),
        ("candidate.actual_student_updates", 1, candidate.get("actual_student_updates")),
        ("schedule", reference.get("schedule"), candidate.get("schedule")),
        ("final_state", reference.get("final_state"), candidate.get("final_state")),
        ("cycle_count", len(reference.get("cycles", [])), len(candidate.get("cycles", []))),
    ):
        exact.append(_exact_gate(label, expected, actual))

    reference_cycles = reference.get("cycles", [])
    candidate_cycles = candidate.get("cycles", [])
    for index, before in enumerate(reference_cycles):
        cycle_label = f"cycle_{index + 1}"
        if index >= len(candidate_cycles):
            exact.append(_exact_gate(f"{cycle_label}.present", True, False))
            continue
        after = candidate_cycles[index]
        exact.append(_exact_gate(f"{cycle_label}.cycle", before.get("cycle"), after.get("cycle")))
        exact.append(_exact_gate(f"{cycle_label}.state", before.get("state"), after.get("state")))
        exact.append(
            _exact_gate(
                f"{cycle_label}.stat_keys",
                sorted(before.get("stats", {})),
                sorted(after.get("stats", {})),
            )
        )
        for key, left in before.get("stats", {}).items():
            right = after.get("stats", {}).get(key)
            if isinstance(left, float) and isinstance(right, (int, float)) and not isinstance(right, bool):
                stats.append(
                    _numeric_gate(
                        f"{cycle_label}.stats.{key}",
                        left,
                        float(right),
                        rtol=stat_rtol,
                        atol=stat_atol,
                    )
                )
            else:
                exact.append(_exact_gate(f"{cycle_label}.stats.{key}", left, right))

    for field in (
        "structure_sha256",
        "serialized_leaf_count",
        "resumed_leaf_count",
        "exact_pickle_round_trip",
        "exact_resume_round_trip",
        "post_resume_update_executed",
    ):
        exact.append(
            _exact_gate(
                f"checkpoint.{field}",
                reference.get("checkpoint", {}).get(field),
                candidate.get("checkpoint", {}).get(field),
            )
        )
    exact.append(
        _exact_gate(
            "initial_checkpoint.structure_sha256",
            reference.get("initial_checkpoint", {}).get("structure_sha256"),
            candidate.get("initial_checkpoint", {}).get("structure_sha256"),
        )
    )
    exact.append(
        _exact_gate(
            "initial_checkpoint.source_sha256",
            protocol["reference"]["initial_checkpoint_sha256"],
            candidate.get("initial_checkpoint", {}).get("source_sha256"),
        )
    )

    initial = _leaf_gates(
        reference["numerical"]["initial"],
        candidate["numerical"]["initial"],
        stage="initial",
        rtol=leaf_rtol,
        atol=leaf_atol,
        require_exact_hash=True,
    )
    final = _leaf_gates(
        reference["numerical"]["final"],
        candidate["numerical"]["final"],
        stage="final",
        rtol=leaf_rtol,
        atol=leaf_atol,
        require_exact_hash=False,
    )
    all_exact = exact + initial["exact"] + final["exact"]
    all_sentinels = initial["sentinels"] + final["sentinels"]
    all_hashes = initial["hashes"] + final["hashes"]
    all_aggregates = initial["aggregates"] + final["aggregates"]
    required_hashes = [gate for gate in all_hashes if gate["required"]]
    failures = [
        gate
        for gate in all_exact + required_hashes + all_sentinels + stats + all_aggregates
        if not gate["pass"]
    ]
    initial_exact_hashes = [
        gate
        for gate in initial["hashes"]
        if gate["pass"]
    ]
    final_exact_hashes = [
        gate
        for gate in final["hashes"]
        if gate["pass"]
    ]
    finite_aggregate_errors = [
        gate for gate in all_aggregates if "absolute_error" in gate
    ]
    return {
        "schema_version": 1,
        "purpose": "exhaustive non-evidence one-update aggregate parity audit",
        "paper_evidence": False,
        "backend": backend,
        "tolerances": tolerances,
        "exact_gates": all_exact,
        "leaf_hash_comparisons": all_hashes,
        "nonfinite_sentinel_gates": all_sentinels,
        "statistic_gates": stats,
        "aggregate_gates": all_aggregates,
        "summary": {
            "status": "passed" if not failures else "failed",
            "failure_count": len(failures),
            "first_failure": None if not failures else failures[0]["label"],
            "exact_gate_count": len(all_exact),
            "nonfinite_sentinel_gate_count": len(all_sentinels),
            "statistic_gate_count": len(stats),
            "aggregate_gate_count": len(all_aggregates),
            "expected_leaf_count_per_stage": 91,
            "initial_exact_leaf_hash_count": len(initial_exact_hashes),
            "final_exact_leaf_hash_count": len(final_exact_hashes),
            "maximum_aggregate_absolute_error": max(
                (gate["absolute_error"] for gate in finite_aggregate_errors), default=0.0
            ),
            "maximum_aggregate_relative_error": max(
                (gate["relative_error"] for gate in finite_aggregate_errors), default=0.0
            ),
        },
        "failures": failures,
    }


def compare_paths(reference_path: Path, candidate_path: Path, *, backend: str) -> dict[str, Any]:
    require(sha256(PROTOCOL) == PROTOCOL_SHA256, "frozen protocol drift")
    require(reference_path.is_file() and not reference_path.is_symlink(), "unsafe reference receipt")
    require(candidate_path.is_file() and not candidate_path.is_symlink(), "unsafe candidate receipt")
    require(sha256(reference_path) == REFERENCE_RECEIPT_SHA256, "reference receipt drift")
    protocol = json.loads(PROTOCOL.read_text())
    report = compare_documents(
        json.loads(reference_path.read_text()),
        json.loads(candidate_path.read_text()),
        backend=backend,
        protocol=protocol,
    )
    report["hashes"] = {
        "protocol_sha256": PROTOCOL_SHA256,
        "reference_receipt_sha256": sha256(reference_path),
        "candidate_receipt_sha256": sha256(candidate_path),
        "auditor_sha256": sha256(Path(__file__).resolve()),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-receipt", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare_paths(
            args.reference_receipt.resolve(),
            args.candidate_receipt.resolve(),
            backend=args.backend,
        )
        atomic_json(args.output.resolve(), report)
    except (AuditError, KeyError, OSError, ValueError, TypeError) as error:
        print(f"HIGHEST_PRECISION_AUDIT_ERROR: {error}", file=os.sys.stderr)
        return 2
    summary = report["summary"]
    print(
        "HIGHEST_PRECISION_AUDIT_"
        f"{summary['status'].upper()} backend={args.backend} "
        f"aggregates={summary['aggregate_gate_count']} "
        f"failures={summary['failure_count']}"
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
