#!/usr/bin/env python3
"""Apply the content-addressed highest-precision LSTM compatibility patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
CONTRACT = ROOT / "PATCH_CONTRACT.json"
PATCH = ROOT / "minimax-highest-lstm.patch"
CONTRACT_SHA256 = "7d8744ff34d064bd324cdc3d92b972b8050f492ff580edc6e44870bbf4aa969e"
PATCH_SHA256 = "a16f4394af0d89289314ab4a11ea43d3334ecba36a22e3c86ed11633d15fb9db"
UPSTREAM_COMMIT = "d053054c5290a04c1c4cd8b55704d999cad73e30"
UPSTREAM_TREE = "b0cace1fc54984e21a842f12d15d0b899e33d270"
FRONTIER_MANIFEST_SHA256 = "d929efa2f059a93125e217ec4713ae81670c769d979c67abd2b10efc64268af3"
MODERN_MANIFEST_SHA256 = "ea5fb73c0072cd95829630344e559f02a83f65b0f8b479845ef4dff8921ff65c"
OUTPUT_MANIFEST = ".blackwell_highest_lstm_overlay.json"


class PatchError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PatchError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(source: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def atomic_json(path: Path, document: Mapping[str, Any]) -> None:
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


def expected_effective_hashes(frontier: Mapping[str, Any], modern: Mapping[str, Any]) -> dict[str, str]:
    expected = dict(frontier["overlay_file_sha256"])
    expected.update(modern["file_sha256"])
    return expected


def apply(source: Path) -> dict[str, Any]:
    source = source.resolve()
    require(source.is_dir() and not source.is_symlink(), "unsafe source directory")
    require(sha256(CONTRACT) == CONTRACT_SHA256, "patch contract drift")
    require(sha256(PATCH) == PATCH_SHA256, "patch file drift")
    contract = json.loads(CONTRACT.read_text())
    require(git(source, "rev-parse", "HEAD") == UPSTREAM_COMMIT, "source commit drift")
    require(git(source, "rev-parse", "HEAD^{tree}") == UPSTREAM_TREE, "source tree drift")
    frontier_path = source / ".frontierrl_overlay.json"
    modern_path = source / ".blackwell_training_overlay.json"
    require(frontier_path.is_file() and not frontier_path.is_symlink(), "bad Frontier manifest")
    require(modern_path.is_file() and not modern_path.is_symlink(), "bad modern manifest")
    require(sha256(frontier_path) == FRONTIER_MANIFEST_SHA256, "Frontier manifest drift")
    require(sha256(modern_path) == MODERN_MANIFEST_SHA256, "modern manifest drift")
    frontier = json.loads(frontier_path.read_text())
    modern = json.loads(modern_path.read_text())
    target_relative = "src/minimax/models/common.py"
    target = source / target_relative
    target_contract = contract["files"][target_relative]
    effective = expected_effective_hashes(frontier, modern)
    require(effective[target_relative] == target_contract["source_sha256"], "parent hash drift")
    for relative, expected_hash in effective.items():
        path = source / relative
        require(path.is_file() and not path.is_symlink(), f"unsafe parent file: {relative}")
        if relative == target_relative:
            require(
                sha256(path) in {
                    target_contract["source_sha256"],
                    target_contract["applied_sha256"],
                },
                "target is neither source nor applied state",
            )
        else:
            require(sha256(path) == expected_hash, f"undeclared parent file drift: {relative}")

    old = """\t\tif self.recurrent_arch == 'lstm':
\t\t\trnn_cell = nn.OptimizedLSTMCell(**rnn_kwargs) # defaults to orth init
\t\telif self.recurrent_arch == 'gru':
\t\t\trnn_cell = nn.GRUCell(**rnn_kwargs)
\t\telse:
\t\t\traise ValueError(f'Unsupported recurrent_arch={self.recurrent_arch}')

\t\tnew_rnn_state, y = rnn_cell(rnn_state, x)
"""
    new = """\t\tif self.recurrent_arch == 'lstm':
\t\t\trnn_cell = nn.OptimizedLSTMCell(**rnn_kwargs) # defaults to orth init
\t\t\twith jax.default_matmul_precision('highest'):
\t\t\t\tnew_rnn_state, y = rnn_cell(rnn_state, x)
\t\telif self.recurrent_arch == 'gru':
\t\t\trnn_cell = nn.GRUCell(**rnn_kwargs)
\t\t\tnew_rnn_state, y = rnn_cell(rnn_state, x)
\t\telse:
\t\t\traise ValueError(f'Unsupported recurrent_arch={self.recurrent_arch}')

"""
    before_hash = sha256(target)
    if before_hash == target_contract["source_sha256"]:
        text = target.read_text()
        require(text.count(old) == 1, "source transformation anchor drift")
        require(text.count(new) == 0, "applied block already present in source state")
        target.write_text(text.replace(old, new, 1))
    require(sha256(target) == target_contract["applied_sha256"], "applied target hash drift")
    applied_text = target.read_text()
    require(
        applied_text.count(target_contract["required_context"])
        == target_contract["applied_lstm_context_count"],
        "precision context count drift",
    )
    require(applied_text.count("nn.OptimizedLSTMCell(**rnn_kwargs)") == 1, "LSTM cell drift")
    require(applied_text.count("nn.GRUCell(**rnn_kwargs)") == 1, "GRU cell drift")
    require(
        sum(path.read_text().count("jax.tree_map") for path in (source / "src/minimax").rglob("*.py"))
        == 0,
        "removed JAX API returned",
    )
    manifest = {
        "schema_version": 1,
        "overlay": contract["overlay"],
        "contract_sha256": CONTRACT_SHA256,
        "patch_sha256": PATCH_SHA256,
        "parent_modernization_manifest_sha256": MODERN_MANIFEST_SHA256,
        "parent_frontier_manifest_sha256": FRONTIER_MANIFEST_SHA256,
        "upstream_commit": UPSTREAM_COMMIT,
        "file_sha256": {target_relative: target_contract["applied_sha256"]},
        "precision": "highest",
        "scope": "OptimizedLSTMCell call only",
        "paper_evidence": False,
    }
    atomic_json(source / OUTPUT_MANIFEST, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = apply(args.source)
    except (PatchError, KeyError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"HIGHEST_PRECISION_PATCH_ERROR: {error}", file=os.sys.stderr)
        return 1
    print(
        "HIGHEST_PRECISION_PATCH_OK "
        f"contract={manifest['contract_sha256']} file_count={len(manifest['file_sha256'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
