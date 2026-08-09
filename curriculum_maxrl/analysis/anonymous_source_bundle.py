"""Build and verify the deterministic anonymous standalone source bundle.

The tracked scope is a fixed allowlist.  Build mode reads only those files and
three explicitly receipted historical witnesses; it never walks the repository.
Anonymization changes are recorded as original/export SHA-256 pairs so frozen
scientific hash chains continue to refer to the original bytes.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import gzip
import hashlib
import importlib
import io
import json
import re
import sys
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


SCOPE_SCHEMA = "curriculum-maxrl/anonymous-source-bundle-scope/v2"
RECEIPT_SCHEMA = "curriculum-maxrl/anonymous-source-bundle-receipt/v2"
DEFAULT_SCOPE = "anonymous_source_bundle_scope.json"
DEFAULT_OUTPUT = "tmp/anonymous-source-bundle/tree"
DEFAULT_ARCHIVE = "tmp/anonymous-source-bundle/curriculum-maxrl-anonymous.tar.gz"
RECEIPT_NAME = "ANONYMIZATION_RECEIPT.json"
GITHUB_OWNER_URL = re.compile(
    r"https://github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)"
)
IDENTITY_TOKEN = re.compile(r"[A-Za-z0-9_-]+")
WINDOWS_USER_PATH = re.compile(
    r"(?i)(?:^|[^A-Za-z0-9])[A-Z]:\\" + "Users" + r"\\"
)
WINDOWS_UNC_PATH = re.compile(r"(?i)(?:^|[^\\])\\\\[^\\\s]+\\[^\\\s]+")
HEX64 = re.compile(r"[0-9a-f]{64}")
PROCURL_CONFIRMATORY_RAW = (
    "frontier_rl/examples/acrobot_procurl_selection_confirmatory.json"
)
PROCURL_DEVELOPMENT_RAW = (
    "frontier_rl/examples/acrobot_procurl_selection_development.json"
)
NEVER_READ_GLOBS = (
    PROCURL_CONFIRMATORY_RAW,
    PROCURL_DEVELOPMENT_RAW,
    "frontier_rl/examples/acrobot_procurl_selection_quick*.json",
)
PROCURL_EXTERNAL_MANIFEST = (
    "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_EXTERNAL_RAW_MANIFEST.json"
)
PROCURL_LOCK = "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_LOCK.json"
PROCURL_GATE = (
    "frontier_rl/examples/acrobot_procurl_selection_development_gates.json"
)
PROCURL_ANALYSIS = (
    "frontier_rl/examples/acrobot_procurl_selection_analysis.json"
)
PROCURL_PORTABLE = (
    "frontier_rl/examples/acrobot_procurl_selection_portable_verification.json"
)
PROCURL_DIAGNOSTICS = (
    "frontier_rl/examples/acrobot_procurl_selection_diagnostics.json"
)
PROCURL_RESULTS = "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_RESULTS.md"
PROCURL_RAW_SIZE = 1_374_886_097
PROCURL_RAW_SHA256 = (
    "b1f8756c249effab8c77101c8bca73ddf708a5e143c18fe8742fd5712fdd7c12"
)
PROCURL_DEVELOPMENT_SIZE = 11_453_535
PROCURL_DEVELOPMENT_SHA256 = (
    "6d9fa639295e35cd8a8da810ace82d330c863edace702db4e3f7d25a9ad82ba8"
)
PROCURL_SAFE_COMPACT_PATHS = (
    "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_EXTERNAL_RAW_MANIFEST.json",
    "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_LOCK.json",
    "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_PROTOCOL.md",
    "frontier_rl/examples/ACROBOT_PROCURL_SELECTION_RESULTS.md",
    "frontier_rl/examples/PROCURL_PRIMARY_SOURCE_PROVENANCE.md",
    "frontier_rl/examples/acrobot_procurl_selection_analysis.json",
    "frontier_rl/examples/acrobot_procurl_selection_development_gates.json",
    "frontier_rl/examples/acrobot_procurl_selection_diagnostics.json",
    "frontier_rl/examples/acrobot_procurl_selection_portable_verification.json",
    "frontier_rl/examples/analyze_acrobot_procurl_selection.py",
    "frontier_rl/examples/build_acrobot_procurl_external_manifest.py",
    "frontier_rl/examples/build_acrobot_procurl_selection_lock.py",
    "frontier_rl/examples/extract_acrobot_procurl_selection_diagnostics.py",
    "frontier_rl/examples/run_acrobot_procurl_selection.py",
    "frontier_rl/examples/test_analyze_acrobot_procurl_selection.py",
    "frontier_rl/examples/test_build_acrobot_procurl_external_manifest.py",
    "frontier_rl/examples/test_build_acrobot_procurl_selection_lock.py",
    "frontier_rl/examples/test_extract_acrobot_procurl_selection_diagnostics.py",
    "frontier_rl/examples/test_run_acrobot_procurl_selection.py",
    "frontier_rl/examples/test_verify_acrobot_procurl_selection_portable.py",
    "frontier_rl/examples/verify_acrobot_procurl_selection_portable.py",
)
EXPECTED_FORBIDDEN_PREFIXES = tuple(
    "/" + suffix
    for suffix in (
        "Users/",
        "home/",
        "tmp/",
        "private/" + "tmp/",
        "var/folders/",
        "root/",
        "Volumes/",
    )
)
EXPECTED_FORBIDDEN_LITERALS = ("file" + "://",)
EXPECTED_IDENTITY_SHA256 = (
    "715e3d85f48fd4c1cf28e434f011c4d858138eb55242384cc3d90733f1c86366",
)


class BundleError(RuntimeError):
    """The requested bundle is unsafe, stale, incomplete, or modified."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _projected_json_bytes(value: Any) -> bytes:
    """Serialize a transformed artifact without reordering semantic ledgers."""
    return (
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise BundleError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except BundleError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot parse strict JSON {label}: {exc}") from exc


def _load_json(path: Path, label: str | None = None) -> Any:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise BundleError(f"cannot read {label or path}: {exc}") from exc
    return _parse_json(payload, label or str(path))


def _safe_relative(raw: object) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise BundleError(f"unsafe repository-relative path: {raw!r}")
    if raw.startswith("/") or any(part in {"", ".", ".."} for part in raw.split("/")):
        raise BundleError(f"unsafe repository-relative path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or path.as_posix() != raw:
        raise BundleError(f"noncanonical repository-relative path: {raw!r}")
    return raw


def _assert_no_symlink_components(root: Path, relative: str, label: str) -> Path:
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise BundleError(f"{label} has a symlink component: {relative}")
    try:
        resolved = cursor.resolve(strict=True)
    except OSError as exc:
        raise BundleError(f"{label} is missing or unreadable: {relative}") from exc
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise BundleError(f"{label} resolves outside its root: {relative}") from exc
    if not resolved.is_file():
        raise BundleError(f"{label} is not a regular file: {relative}")
    return resolved


def _assert_config_inside(repo_root: Path, scope_path: Path) -> tuple[Path, str]:
    lexical_root = repo_root.absolute()
    lexical_scope = scope_path.absolute()
    try:
        relative = lexical_scope.relative_to(lexical_root).as_posix()
    except ValueError as exc:
        raise BundleError("scope config must be inside the repository") from exc
    _safe_relative(relative)
    path = _assert_no_symlink_components(lexical_root, relative, "scope config")
    return path, relative


@dataclass(frozen=True)
class ContentPolicy:
    prefixes: tuple[str, ...]
    literals: tuple[str, ...]
    identity_sha256: frozenset[str]

    def contains(self, value: str) -> bool:
        folded = value.casefold()
        return (
            any(prefix in value for prefix in self.prefixes)
            or any(literal.casefold() in folded for literal in self.literals)
            or any(
                _sha256(match.group(0).casefold().encode("utf-8"))
                in self.identity_sha256
                for match in IDENTITY_TOKEN.finditer(value)
            )
            or WINDOWS_USER_PATH.search(value) is not None
            or WINDOWS_UNC_PATH.search(value) is not None
        )


def _policy(scope_or_receipt: dict[str, Any]) -> ContentPolicy:
    raw = scope_or_receipt["content_policy"]
    prefixes = raw.get("forbidden_prefixes")
    literals = raw.get("forbidden_literals")
    identities = raw.get("forbidden_identity_sha256")
    if not isinstance(prefixes, list) or not prefixes:
        raise BundleError("content policy needs forbidden prefixes")
    if not isinstance(literals, list) or not isinstance(identities, list):
        raise BundleError("invalid content policy lists")
    if any(not isinstance(value, str) or not value for value in prefixes + literals):
        raise BundleError("empty or non-string content-policy entry")
    if tuple(prefixes) != EXPECTED_FORBIDDEN_PREFIXES:
        raise BundleError("content policy forbidden-prefix inventory differs")
    if tuple(literals) != EXPECTED_FORBIDDEN_LITERALS:
        raise BundleError("content policy forbidden-literal inventory differs")
    if tuple(identities) != EXPECTED_IDENTITY_SHA256:
        raise BundleError("content policy forbidden-identity inventory differs")
    if any(not isinstance(value, str) or HEX64.fullmatch(value) is None for value in identities):
        raise BundleError("invalid forbidden identity digest")
    return ContentPolicy(tuple(prefixes), tuple(literals), frozenset(identities))


def _path_is_forbidden(relative: str, patterns: list[str]) -> str | None:
    folded = relative.casefold()
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern:
            raise BundleError("invalid forbidden-selection glob")
        if fnmatch.fnmatchcase(folded, pattern.casefold()):
            return pattern
    return None


def _assert_not_never_read(relative: str) -> None:
    pattern = _path_is_forbidden(relative, list(NEVER_READ_GLOBS))
    if pattern:
        raise BundleError(
            f"path is forbidden before any read by the sealed-artifact policy: "
            f"{relative} ({pattern})"
        )


def _validate_scope(scope: Any) -> dict[str, Any]:
    if not isinstance(scope, dict) or scope.get("schema") != SCOPE_SCHEMA:
        raise BundleError("unsupported anonymous source-bundle scope schema")
    expected_fields = {
        "schema",
        "expected_selected_file_count",
        "archive_root",
        "groups",
        "unshipped_artifacts",
        "omissions",
        "forbidden_selection_globs",
        "content_policy",
        "approved_json_transforms",
        "approved_text_transforms",
        "executable_paths",
    }
    if set(scope) != expected_fields:
        raise BundleError(
            f"scope fields differ: missing={sorted(expected_fields - set(scope))}, "
            f"unexpected={sorted(set(scope) - expected_fields)}"
        )
    if not isinstance(scope["groups"], list) or not scope["groups"]:
        raise BundleError("scope needs at least one fixed allowlist group")
    if not isinstance(scope["forbidden_selection_globs"], list):
        raise BundleError("forbidden_selection_globs must be a list")
    if not isinstance(scope["archive_root"], str) or not scope["archive_root"]:
        raise BundleError("archive_root must be nonempty")
    _safe_relative(scope["archive_root"])
    _policy(scope)

    paths: set[str] = set()
    names: set[str] = set()
    for group in scope["groups"]:
        if not isinstance(group, dict) or set(group) != {
            "name",
            "expected_file_count",
            "paths",
        }:
            raise BundleError("each allowlist group needs name/count/paths only")
        name = group["name"]
        if not isinstance(name, str) or not name or name in names:
            raise BundleError(f"invalid or duplicate allowlist group: {name!r}")
        names.add(name)
        if not isinstance(group["paths"], list):
            raise BundleError(f"group paths must be a list: {name}")
        normalized = [_safe_relative(value) for value in group["paths"]]
        if normalized != sorted(normalized):
            raise BundleError(f"allowlist group is not bytewise sorted: {name}")
        if len(normalized) != group["expected_file_count"]:
            raise BundleError(f"allowlist group count is stale: {name}")
        duplicate = paths.intersection(normalized)
        if duplicate:
            raise BundleError(f"duplicate selected path across groups: {sorted(duplicate)}")
        for relative in normalized:
            _assert_not_never_read(relative)
            pattern = _path_is_forbidden(relative, scope["forbidden_selection_globs"])
            if pattern:
                raise BundleError(
                    f"selected path is forbidden before any read: {relative} ({pattern})"
                )
        paths.update(normalized)
    if len(paths) != scope["expected_selected_file_count"]:
        raise BundleError(
            f"selected file count is stale: {len(paths)} != "
            f"{scope['expected_selected_file_count']}"
        )
    if RECEIPT_NAME in paths:
        raise BundleError("generated receipt cannot be a selected source file")

    unshipped = scope["unshipped_artifacts"]
    if not isinstance(unshipped, list):
        raise BundleError("unshipped_artifacts must be a list")
    unshipped_paths: set[str] = set()
    previous_unshipped = ""
    for artifact in unshipped:
        expected = {
            "path",
            "original_bytes",
            "original_sha256",
            "availability",
            "content_addressed_download_uri",
            "reason",
        }
        if not isinstance(artifact, dict) or set(artifact) != expected:
            raise BundleError(
                "each unshipped artifact needs path/bytes/SHA/availability/URI/reason"
            )
        relative = _safe_relative(artifact["path"])
        if relative <= previous_unshipped:
            raise BundleError("unshipped artifact inventory is not bytewise sorted")
        previous_unshipped = relative
        if relative in paths or relative in unshipped_paths:
            raise BundleError(f"duplicate selected/unshipped path: {relative}")
        never_read = _path_is_forbidden(relative, list(NEVER_READ_GLOBS))
        if never_read is None:
            raise BundleError(
                f"unshipped artifact is not protected by the never-read policy: {relative}"
            )
        if _path_is_forbidden(relative, scope["forbidden_selection_globs"]) is None:
            raise BundleError(
                f"unshipped artifact is not covered by the exclusion policy: {relative}"
            )
        if (
            type(artifact["original_bytes"]) is not int
            or artifact["original_bytes"] <= 0
        ):
            raise BundleError(f"invalid unshipped byte count: {relative}")
        if HEX64.fullmatch(str(artifact["original_sha256"])) is None:
            raise BundleError(f"invalid unshipped SHA-256: {relative}")
        if not isinstance(artifact["availability"], str) or not artifact["availability"]:
            raise BundleError(f"missing unshipped availability: {relative}")
        if artifact["content_addressed_download_uri"] is not None:
            raise BundleError(
                f"unshipped artifact unexpectedly advertises a download URI: {relative}"
            )
        if not isinstance(artifact["reason"], str) or not artifact["reason"]:
            raise BundleError(f"missing unshipped reason: {relative}")
        unshipped_paths.add(relative)

    omissions = scope["omissions"]
    if not isinstance(omissions, list):
        raise BundleError("omissions must be a list")
    omitted_paths: set[str] = set()
    for omission in omissions:
        if not isinstance(omission, dict) or set(omission) != {
            "path",
            "original_bytes",
            "original_sha256",
            "reason",
        }:
            raise BundleError("each omission needs path/bytes/SHA/reason only")
        relative = _safe_relative(omission["path"])
        _assert_not_never_read(relative)
        if relative in paths or relative in unshipped_paths or relative in omitted_paths:
            raise BundleError(f"duplicate selected/omitted path: {relative}")
        if not isinstance(omission["original_bytes"], int) or omission["original_bytes"] < 0:
            raise BundleError(f"invalid omission byte count: {relative}")
        if HEX64.fullmatch(str(omission["original_sha256"])) is None:
            raise BundleError(f"invalid omission SHA-256: {relative}")
        if not isinstance(omission["reason"], str) or not omission["reason"]:
            raise BundleError(f"missing omission reason: {relative}")
        if _path_is_forbidden(relative, scope["forbidden_selection_globs"]) is None:
            raise BundleError(f"omission is not covered by the exclusion policy: {relative}")
        if not fnmatch.fnmatchcase(relative.casefold(), "*aborted*"):
            raise BundleError(f"only explicitly aborted historical witnesses may be omitted: {relative}")
        omitted_paths.add(relative)

    for key in (
        "approved_json_transforms",
        "approved_text_transforms",
        "executable_paths",
    ):
        if not isinstance(scope[key], list):
            raise BundleError(f"scope field must be a list: {key}")
    executables = [_safe_relative(value) for value in scope["executable_paths"]]
    if len(executables) != len(set(executables)) or not set(executables) <= paths:
        raise BundleError("executable_paths are duplicate or outside the allowlist")
    return scope


def load_scope(repo_root: Path, scope_path: Path) -> tuple[dict[str, Any], str, bytes]:
    real_path, relative = _assert_config_inside(repo_root, scope_path)
    payload = real_path.read_bytes()
    scope = _validate_scope(_parse_json(payload, relative))
    return scope, relative, payload


def _selected_by_group(scope: dict[str, Any]) -> list[tuple[str, str]]:
    return sorted(
        (relative, group["name"])
        for group in scope["groups"]
        for relative in group["paths"]
    )


def _pointer_part(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _walk_json_strings(value: Any, pointer: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield pointer, value
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json_strings(child, f"{pointer}/{index}")
    elif isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise BundleError(f"non-string JSON key at {pointer or '/'}")
            yield from _walk_json_strings(child, f"{pointer}/{_pointer_part(key)}")


def _assert_json_keys_clean(value: Any, policy: ContentPolicy, pointer: str = "") -> None:
    if isinstance(value, list):
        for index, child in enumerate(value):
            _assert_json_keys_clean(child, policy, f"{pointer}/{index}")
    elif isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise BundleError(f"non-string JSON key at {pointer or '/'}")
            if policy.contains(key):
                raise BundleError(f"forbidden content in JSON key at {pointer or '/'}")
            _assert_json_keys_clean(child, policy, f"{pointer}/{_pointer_part(key)}")


def _json_pointer_set(value: Any, pointer: str, replacement: str) -> str:
    if not pointer.startswith("/"):
        raise BundleError(f"only non-root JSON pointers are supported: {pointer!r}")
    parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]
    parent = value
    for part in parts[:-1]:
        if isinstance(parent, list):
            try:
                parent = parent[int(part)]
            except (ValueError, IndexError) as exc:
                raise BundleError(f"invalid JSON pointer: {pointer}") from exc
        elif isinstance(parent, dict) and part in parent:
            parent = parent[part]
        else:
            raise BundleError(f"invalid JSON pointer: {pointer}")
    leaf = parts[-1]
    if isinstance(parent, list):
        try:
            index = int(leaf)
            old = parent[index]
        except (ValueError, IndexError) as exc:
            raise BundleError(f"invalid JSON pointer: {pointer}") from exc
        if not isinstance(old, str):
            raise BundleError(f"approved JSON pointer is not a string: {pointer}")
        parent[index] = replacement
        return old
    if isinstance(parent, dict) and leaf in parent:
        old = parent[leaf]
        if not isinstance(old, str):
            raise BundleError(f"approved JSON pointer is not a string: {pointer}")
        parent[leaf] = replacement
        return old
    raise BundleError(f"invalid JSON pointer: {pointer}")


def _json_approvals(
    scope: dict[str, Any], selected: set[str], policy: ContentPolicy
) -> dict[tuple[str, str], dict[str, str]]:
    approvals: dict[tuple[str, str], dict[str, str]] = {}
    for raw in scope["approved_json_transforms"]:
        if not isinstance(raw, dict) or set(raw) != {"file", "json_pointer", "replacement"}:
            raise BundleError("invalid JSON-transform declaration")
        relative = _safe_relative(raw["file"])
        pointer = raw["json_pointer"]
        replacement = raw["replacement"]
        if relative not in selected or not relative.endswith(".json"):
            raise BundleError(f"JSON transform targets an invalid file: {relative}")
        if not isinstance(pointer, str) or not pointer.startswith("/"):
            raise BundleError(f"invalid JSON transform pointer: {relative}#{pointer}")
        if not isinstance(replacement, str) or not replacement or policy.contains(replacement):
            raise BundleError(f"unsafe JSON replacement: {relative}#{pointer}")
        key = (relative, pointer)
        if key in approvals:
            raise BundleError(f"duplicate JSON transform: {relative}#{pointer}")
        approvals[key] = {
            "file": relative,
            "json_pointer": pointer,
            "replacement": replacement,
        }
    return approvals


def _project_json(
    relative: str,
    source: bytes,
    policy: ContentPolicy,
    approvals: dict[tuple[str, str], dict[str, str]],
) -> tuple[bytes, list[dict[str, Any]], set[tuple[str, str]]]:
    original = _parse_json(source, relative)
    _assert_json_keys_clean(original, policy)
    projected = copy.deepcopy(original)
    transforms: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    for pointer, value in _walk_json_strings(original):
        if not policy.contains(value):
            continue
        key = (relative, pointer)
        approval = approvals.get(key)
        if approval is None:
            raise BundleError(f"unapproved forbidden JSON value: {relative}#{pointer or '/'}")
        old = _json_pointer_set(projected, pointer, approval["replacement"])
        if old != value:
            raise BundleError(f"JSON pointer walk/set disagreement: {relative}#{pointer}")
        used.add(key)
        transforms.append(
            {
                "kind": "json_pointer",
                "json_pointer": pointer,
                "original_value_sha256": _sha256(value.encode("utf-8")),
                "replacement": approval["replacement"],
                "occurrence_count": 1,
            }
        )
    for pointer, value in _walk_json_strings(projected):
        if policy.contains(value):
            raise BundleError(f"forbidden JSON content survived: {relative}#{pointer or '/'}")
    return (_projected_json_bytes(projected) if transforms else source), transforms, used


def _validate_jsonl(payload: bytes, relative: str, policy: ContentPolicy) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BundleError(f"selected JSONL file is not UTF-8: {relative}") from exc
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        value = _parse_json(line.encode("utf-8"), f"{relative}:{line_number}")
        _assert_json_keys_clean(value, policy)
        for pointer, string in _walk_json_strings(value):
            if policy.contains(string):
                raise BundleError(
                    f"forbidden JSONL content: {relative}:{line_number}#{pointer or '/'}"
                )


def _text_approvals(
    scope: dict[str, Any], selected: set[str], policy: ContentPolicy
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    allowed_kinds = {"exact", "github_owner", "region"}
    for raw in scope["approved_text_transforms"]:
        if not isinstance(raw, dict):
            raise BundleError("invalid text-transform declaration")
        kind = raw.get("kind")
        relative = _safe_relative(raw.get("file"))
        if kind not in allowed_kinds or relative not in selected or relative.endswith(".json"):
            raise BundleError(f"invalid text transform target/kind: {relative} ({kind})")
        replacement = raw.get("replacement")
        if not isinstance(replacement, str) or not replacement or policy.contains(replacement):
            raise BundleError(f"unsafe text replacement: {relative} ({kind})")
        expected_fields = {
            "exact": {"kind", "file", "match", "replacement", "expected_occurrences", "purpose"},
            "github_owner": {
                "kind",
                "file",
                "github_owner_sha256",
                "replacement",
                "expected_occurrences",
                "purpose",
            },
            "region": {
                "kind",
                "file",
                "start_marker",
                "end_marker",
                "expected_original_sha256",
                "replacement",
                "expected_occurrences",
                "purpose",
            },
        }[kind]
        if set(raw) != expected_fields:
            raise BundleError(f"text-transform fields differ: {relative} ({kind})")
        count = raw["expected_occurrences"]
        if not isinstance(count, int) or count < 1:
            raise BundleError(f"invalid text-transform count: {relative} ({kind})")
        if not isinstance(raw["purpose"], str) or not raw["purpose"]:
            raise BundleError(f"missing text-transform purpose: {relative} ({kind})")
        if kind == "exact":
            selector = raw["match"]
            if not isinstance(selector, str) or not selector:
                raise BundleError(f"invalid exact transform match: {relative}")
        elif kind == "github_owner":
            selector = raw["github_owner_sha256"]
            if HEX64.fullmatch(str(selector)) is None or selector not in policy.identity_sha256:
                raise BundleError(f"invalid GitHub owner digest: {relative}")
        else:
            selector = raw["expected_original_sha256"]
            if HEX64.fullmatch(str(selector)) is None:
                raise BundleError(f"invalid region digest: {relative}")
            if not all(isinstance(raw[key], str) and raw[key] for key in ("start_marker", "end_marker")):
                raise BundleError(f"invalid region markers: {relative}")
        key = (relative, f"{kind}:{selector}")
        if key in seen:
            raise BundleError(f"duplicate text transform: {key}")
        seen.add(key)
        result.setdefault(relative, []).append(raw)
    return result


def _project_text(
    relative: str,
    source: bytes,
    policy: ContentPolicy,
    approvals: list[dict[str, Any]],
) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        original = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BundleError(f"selected text file is not UTF-8: {relative}") from exc
    projected = original
    transforms: list[dict[str, Any]] = []
    for approval in approvals:
        kind = approval["kind"]
        if kind == "exact":
            match = approval["match"]
            actual = original.count(match)
            if actual != approval["expected_occurrences"]:
                raise BundleError(
                    f"stale exact transform count: {relative}: {actual} != "
                    f"{approval['expected_occurrences']}"
                )
            if projected.count(match) != actual:
                raise BundleError(f"overlapping exact transforms are forbidden: {relative}")
            projected = projected.replace(match, approval["replacement"])
            digest = _sha256(match.encode("utf-8"))
        elif kind == "github_owner":
            matches = {
                candidate.group(0)
                for candidate in GITHUB_OWNER_URL.finditer(original)
                if _sha256(candidate.group(1).casefold().encode("utf-8"))
                == approval["github_owner_sha256"]
            }
            if len(matches) != 1:
                raise BundleError(
                    f"stale or ambiguous GitHub-owner transform: {relative}: {len(matches)} bases"
                )
            match = next(iter(matches))
            actual = original.count(match)
            if actual != approval["expected_occurrences"]:
                raise BundleError(f"stale GitHub-owner count: {relative}")
            if projected.count(match) != actual:
                raise BundleError(f"overlapping GitHub-owner transforms are forbidden: {relative}")
            projected = projected.replace(match, approval["replacement"])
            digest = _sha256(match.encode("utf-8"))
        else:
            start_marker = approval["start_marker"]
            end_marker = approval["end_marker"]
            starts = [m.start() for m in re.finditer(re.escape(start_marker), original)]
            if len(starts) != approval["expected_occurrences"]:
                raise BundleError(f"stale region start count: {relative}")
            regions: list[str] = []
            for start in starts:
                end_start = original.find(end_marker, start + len(start_marker))
                if end_start < 0:
                    raise BundleError(f"region end marker is missing: {relative}")
                regions.append(original[start : end_start + len(end_marker)])
            if len(set(regions)) != 1 or len(regions) != 1:
                raise BundleError(f"region transform must identify one exact region: {relative}")
            match = regions[0]
            digest = _sha256(match.encode("utf-8"))
            if digest != approval["expected_original_sha256"]:
                raise BundleError(f"stale region digest: {relative}: {digest}")
            if projected.count(match) != 1:
                raise BundleError(f"overlapping region transforms are forbidden: {relative}")
            projected = projected.replace(match, approval["replacement"], 1)
            actual = 1
        transforms.append(
            {
                "kind": f"text_{kind}",
                "purpose": approval["purpose"],
                "original_value_sha256": digest,
                "replacement": approval["replacement"],
                "occurrence_count": actual,
            }
        )
    if policy.contains(projected):
        raise BundleError(f"unapproved forbidden content in text file: {relative}")
    return projected.encode("utf-8"), transforms


def build_bundle(
    repo_root: Path, scope_path: Path
) -> tuple[dict[str, bytes], dict[str, Any], set[str]]:
    lexical_root = repo_root.absolute()
    scope, scope_relative, scope_payload = load_scope(lexical_root, scope_path)
    repo_root = lexical_root.resolve()
    selected_with_groups = _selected_by_group(scope)
    selected = {relative for relative, _ in selected_with_groups}
    policy = _policy(scope)
    json_approvals = _json_approvals(scope, selected, policy)
    text_approvals = _text_approvals(scope, selected, policy)
    used_json: set[tuple[str, str]] = set()
    projected: dict[str, bytes] = {}
    records: list[dict[str, Any]] = []
    transformation_records: list[dict[str, Any]] = []

    for relative, group in selected_with_groups:
        path = _assert_no_symlink_components(repo_root, relative, "selected source")
        source = path.read_bytes()
        if b"\x00" in source:
            raise BundleError(f"NUL byte in selected source file: {relative}")
        if relative.endswith(".json"):
            export, transforms, used = _project_json(
                relative, source, policy, json_approvals
            )
            used_json.update(used)
        else:
            if relative.endswith(".jsonl"):
                _validate_jsonl(source, relative, policy)
            export, transforms = _project_text(
                relative, source, policy, text_approvals.get(relative, [])
            )
        source_sha = _sha256(source)
        export_sha = _sha256(export)
        projected[relative] = export
        records.append(
            {
                "path": relative,
                "group": group,
                "original_bytes": len(source),
                "original_sha256": source_sha,
                "export_bytes": len(export),
                "export_sha256": export_sha,
                "transformation_count": len(transforms),
                "transformation_occurrence_count": sum(
                    item["occurrence_count"] for item in transforms
                ),
            }
        )
        for transform in transforms:
            transformation_records.append(
                {
                    "file": relative,
                    "original_file_sha256": source_sha,
                    "export_file_sha256": export_sha,
                    **transform,
                }
            )
    unused_json = set(json_approvals) - used_json
    if unused_json:
        raise BundleError(f"stale JSON transform(s): {sorted(unused_json)}")
    declared_text = sum(len(items) for items in text_approvals.values())
    observed_text = sum(
        1 for item in transformation_records if str(item["kind"]).startswith("text_")
    )
    if declared_text != observed_text:
        raise BundleError(
            f"stale text transform inventory: {observed_text} != {declared_text}"
        )

    omission_records: list[dict[str, Any]] = []
    for declared in sorted(scope["omissions"], key=lambda item: item["path"]):
        relative = declared["path"]
        path = _assert_no_symlink_components(repo_root, relative, "omitted witness")
        payload = path.read_bytes()
        if len(payload) != declared["original_bytes"] or _sha256(payload) != declared["original_sha256"]:
            raise BundleError(f"omitted witness no longer matches its frozen receipt: {relative}")
        omission_records.append(dict(declared))
    unshipped_records = [dict(record) for record in scope["unshipped_artifacts"]]

    index_lines = [
        f"{record['path']}\0{record['original_sha256']}\0{record['export_sha256']}"
        for record in records
    ]
    group_counts = {
        group["name"]: group["expected_file_count"] for group in scope["groups"]
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "deterministic": True,
        "scope_config": scope_relative,
        "scope_config_sha256": _sha256(scope_payload),
        "archive_root": scope["archive_root"],
        "bundle_index_sha256": _sha256("\n".join(index_lines).encode("utf-8")),
        "summary": {
            "selected_file_count": len(records),
            "generated_receipt_count": 1,
            "archive_file_count": len(records) + 1,
            "group_file_counts": group_counts,
            "transformed_file_count": len(
                {item["file"] for item in transformation_records}
            ),
            "transformation_rule_count": len(transformation_records),
            "transformation_occurrence_count": sum(
                item["occurrence_count"] for item in transformation_records
            ),
            "omitted_witness_count": len(omission_records),
            "unshipped_artifact_count": len(unshipped_records),
            "unshipped_artifact_bytes": sum(
                record["original_bytes"] for record in unshipped_records
            ),
            "unshipped_artifact_uri_count": sum(
                record["content_addressed_download_uri"] is not None
                for record in unshipped_records
            ),
            "unapproved_leak_count": 0,
        },
        "forbidden_selection_globs": scope["forbidden_selection_globs"],
        "content_policy": scope["content_policy"],
        "executable_paths": sorted(scope["executable_paths"]),
        "files": records,
        "transformations": transformation_records,
        "omissions": omission_records,
        "unshipped_artifacts": unshipped_records,
        "scientific_hash_policy": (
            "Frozen scientific records retain original SHA-256 values.  The receipt "
            "binds each original digest to the projected export digest; verifiers "
            "must not relabel projected bytes as the original bytes."
        ),
    }
    _assert_receipt_metadata_clean(receipt, policy)
    return projected, receipt, set(scope["executable_paths"])


def _receipt_maps(receipt: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(receipt, dict) or receipt.get("schema") != RECEIPT_SCHEMA:
        raise BundleError("unsupported or missing source-bundle receipt")
    files = receipt.get("files")
    omissions = receipt.get("omissions")
    if not isinstance(files, list) or not isinstance(omissions, list):
        raise BundleError("receipt files/omissions must be lists")
    file_map: dict[str, dict[str, Any]] = {}
    for record in files:
        if not isinstance(record, dict):
            raise BundleError("receipt file record is not an object")
        relative = _safe_relative(record.get("path"))
        if relative in file_map:
            raise BundleError(f"duplicate receipt file path: {relative}")
        for key in ("original_sha256", "export_sha256"):
            if HEX64.fullmatch(str(record.get(key))) is None:
                raise BundleError(f"invalid {key} for {relative}")
        for key in ("original_bytes", "export_bytes", "transformation_count"):
            if not isinstance(record.get(key), int) or record[key] < 0:
                raise BundleError(f"invalid {key} for {relative}")
        file_map[relative] = record
    omission_map: dict[str, dict[str, Any]] = {}
    for record in omissions:
        if not isinstance(record, dict):
            raise BundleError("receipt omission is not an object")
        relative = _safe_relative(record.get("path"))
        if relative in omission_map or relative in file_map:
            raise BundleError(f"duplicate receipt path: {relative}")
        if HEX64.fullmatch(str(record.get("original_sha256"))) is None:
            raise BundleError(f"invalid omission digest: {relative}")
        if not isinstance(record.get("original_bytes"), int) or record["original_bytes"] < 0:
            raise BundleError(f"invalid omission byte count: {relative}")
        if not isinstance(record.get("reason"), str) or not record["reason"]:
            raise BundleError(f"missing omission reason: {relative}")
        omission_map[relative] = record
    return file_map, omission_map


def _unshipped_map(
    receipt: dict[str, Any],
    file_map: dict[str, dict[str, Any]],
    omission_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    records = receipt.get("unshipped_artifacts")
    if not isinstance(records, list):
        raise BundleError("receipt unshipped_artifacts must be a list")
    result: dict[str, dict[str, Any]] = {}
    previous = ""
    expected_keys = {
        "path",
        "original_bytes",
        "original_sha256",
        "availability",
        "content_addressed_download_uri",
        "reason",
    }
    for record in records:
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise BundleError("receipt unshipped-artifact record is malformed")
        relative = _safe_relative(record["path"])
        if relative <= previous:
            raise BundleError("receipt unshipped-artifact inventory is not sorted")
        previous = relative
        if relative in file_map or relative in omission_map or relative in result:
            raise BundleError(f"duplicate receipt path: {relative}")
        if _path_is_forbidden(relative, list(NEVER_READ_GLOBS)) is None:
            raise BundleError(
                f"receipt unshipped path is outside the never-read policy: {relative}"
            )
        if (
            type(record["original_bytes"]) is not int
            or record["original_bytes"] <= 0
        ):
            raise BundleError(f"invalid unshipped byte count: {relative}")
        if HEX64.fullmatch(str(record["original_sha256"])) is None:
            raise BundleError(f"invalid unshipped digest: {relative}")
        if not isinstance(record["availability"], str) or not record["availability"]:
            raise BundleError(f"missing unshipped availability: {relative}")
        if record["content_addressed_download_uri"] is not None:
            raise BundleError(
                f"receipt unshipped artifact advertises a download URI: {relative}"
            )
        if not isinstance(record["reason"], str) or not record["reason"]:
            raise BundleError(f"missing unshipped reason: {relative}")
        result[relative] = record
    return result


def _verify_receipt_accounting(
    receipt: dict[str, Any],
    file_map: dict[str, dict[str, Any]],
    omission_map: dict[str, dict[str, Any]],
    unshipped_map: dict[str, dict[str, Any]],
) -> None:
    transformations = receipt.get("transformations")
    summary = receipt.get("summary")
    if not isinstance(transformations, list) or not isinstance(summary, dict):
        raise BundleError("receipt transformations/summary are malformed")
    _safe_relative(receipt.get("scope_config"))
    _safe_relative(receipt.get("archive_root"))
    patterns = receipt.get("forbidden_selection_globs")
    if not isinstance(patterns, list):
        raise BundleError("receipt exclusion policy is malformed")
    by_file: dict[str, list[dict[str, Any]]] = {}
    occurrence_count = 0
    for transform in transformations:
        if not isinstance(transform, dict):
            raise BundleError("receipt transformation is not an object")
        relative = _safe_relative(transform.get("file"))
        if relative not in file_map:
            raise BundleError(f"receipt transform targets an unshipped file: {relative}")
        record = file_map[relative]
        if (
            transform.get("original_file_sha256") != record["original_sha256"]
            or transform.get("export_file_sha256") != record["export_sha256"]
        ):
            raise BundleError(f"receipt transform/file digest binding differs: {relative}")
        occurrences = transform.get("occurrence_count")
        if not isinstance(occurrences, int) or occurrences < 1:
            raise BundleError(f"invalid transform occurrence count: {relative}")
        if HEX64.fullmatch(str(transform.get("original_value_sha256"))) is None:
            raise BundleError(f"invalid transformed-value digest: {relative}")
        occurrence_count += occurrences
        by_file.setdefault(relative, []).append(transform)
    for relative, record in file_map.items():
        if len(by_file.get(relative, [])) != record["transformation_count"]:
            raise BundleError(f"receipt transformation inventory differs: {relative}")
        expected_occurrences = sum(
            item["occurrence_count"] for item in by_file.get(relative, [])
        )
        if expected_occurrences != record.get("transformation_occurrence_count"):
            raise BundleError(f"receipt transformation occurrences differ: {relative}")
    for relative in omission_map:
        _assert_not_never_read(relative)
        if _path_is_forbidden(relative, patterns) is None:
            raise BundleError(f"receipt omission is outside the exclusion policy: {relative}")
        if not fnmatch.fnmatchcase(relative.casefold(), "*aborted*"):
            raise BundleError(
                f"receipt omission is not an explicitly aborted historical witness: {relative}"
            )
    for relative in unshipped_map:
        if _path_is_forbidden(relative, patterns) is None:
            raise BundleError(
                f"receipt unshipped artifact is outside the exclusion policy: {relative}"
            )
    group_counts: dict[str, int] = {}
    for record in file_map.values():
        group = record.get("group")
        if not isinstance(group, str) or not group:
            raise BundleError("receipt file lacks a group")
        group_counts[group] = group_counts.get(group, 0) + 1
    if summary.get("group_file_counts") != dict(sorted(group_counts.items())):
        raise BundleError("receipt group counts are stale")
    expected_summary = {
        "selected_file_count": len(file_map),
        "generated_receipt_count": 1,
        "archive_file_count": len(file_map) + 1,
        "transformed_file_count": len(by_file),
        "transformation_rule_count": len(transformations),
        "transformation_occurrence_count": occurrence_count,
        "omitted_witness_count": len(omission_map),
        "unshipped_artifact_count": len(unshipped_map),
        "unshipped_artifact_bytes": sum(
            record["original_bytes"] for record in unshipped_map.values()
        ),
        "unshipped_artifact_uri_count": sum(
            record["content_addressed_download_uri"] is not None
            for record in unshipped_map.values()
        ),
        "unapproved_leak_count": 0,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise BundleError(f"receipt summary differs for {key}: {summary.get(key)!r} != {expected!r}")


def _assert_receipt_metadata_clean(receipt: dict[str, Any], policy: ContentPolicy) -> None:
    metadata = copy.deepcopy(receipt)
    metadata.pop("content_policy", None)
    _assert_json_keys_clean(metadata, policy)
    for pointer, value in _walk_json_strings(metadata):
        if policy.contains(value):
            raise BundleError(f"forbidden content in generated receipt metadata: {pointer or '/'}")


def _scan_export_payload(relative: str, payload: bytes, policy: ContentPolicy) -> None:
    if b"\x00" in payload:
        raise BundleError(f"NUL byte in exported source file: {relative}")
    if relative.endswith(".json"):
        value = _parse_json(payload, relative)
        _assert_json_keys_clean(value, policy)
        for pointer, text in _walk_json_strings(value):
            if policy.contains(text):
                raise BundleError(f"forbidden JSON content in export: {relative}#{pointer or '/'}")
        return
    if relative.endswith(".jsonl"):
        _validate_jsonl(payload, relative, policy)
        return
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BundleError(f"exported source file is not UTF-8: {relative}") from exc
    if policy.contains(text):
        raise BundleError(f"forbidden content in exported text file: {relative}")


def verify_export(bundle_root: Path) -> dict[str, Any]:
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise BundleError(f"bundle root is missing or a symlink: {bundle_root}")
    receipt_path = bundle_root / RECEIPT_NAME
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise BundleError("generated receipt is missing or a symlink")
    receipt = _load_json(receipt_path, RECEIPT_NAME)
    file_map, omission_map = _receipt_maps(receipt)
    unshipped_map = _unshipped_map(receipt, file_map, omission_map)
    _verify_receipt_accounting(receipt, file_map, omission_map, unshipped_map)
    patterns = receipt.get("forbidden_selection_globs")
    if not isinstance(patterns, list):
        raise BundleError("receipt lacks forbidden-selection policy")
    policy = _policy(receipt)
    _assert_receipt_metadata_clean(receipt, policy)

    actual_paths: set[str] = set()
    for path in bundle_root.rglob("*"):
        if path.is_symlink():
            raise BundleError(f"symlink in exported bundle: {path}")
        if path.is_file():
            actual_paths.add(path.relative_to(bundle_root).as_posix())
    expected_paths = set(file_map) | {RECEIPT_NAME}
    if actual_paths != expected_paths:
        raise BundleError(
            f"bundle file set mismatch; missing={sorted(expected_paths - actual_paths)}, "
            f"unexpected={sorted(actual_paths - expected_paths)}"
        )
    index_lines: list[str] = []
    executable_paths = set(receipt.get("executable_paths", []))
    if any(_safe_relative(value) not in file_map for value in executable_paths):
        raise BundleError("receipt executable path is outside the payload")
    for relative, record in sorted(file_map.items()):
        pattern = _path_is_forbidden(relative, patterns)
        if pattern:
            raise BundleError(f"forbidden path in export: {relative} ({pattern})")
        path = _assert_no_symlink_components(bundle_root, relative, "exported source")
        payload = path.read_bytes()
        if len(payload) != record["export_bytes"] or _sha256(payload) != record["export_sha256"]:
            raise BundleError(f"exported source differs from receipt: {relative}")
        transformed = record["transformation_count"] > 0
        hashes_differ = record["original_sha256"] != record["export_sha256"]
        if transformed != hashes_differ:
            raise BundleError(f"receipt transformation/hash relation is inconsistent: {relative}")
        _scan_export_payload(relative, payload, policy)
        index_lines.append(
            f"{relative}\0{record['original_sha256']}\0{record['export_sha256']}"
        )
    for relative in omission_map:
        if (bundle_root / relative).exists() or (bundle_root / relative).is_symlink():
            raise BundleError(f"omitted witness unexpectedly shipped: {relative}")
    for relative in unshipped_map:
        if (bundle_root / relative).exists() or (bundle_root / relative).is_symlink():
            raise BundleError(f"unshipped artifact unexpectedly shipped: {relative}")
    index = _sha256("\n".join(index_lines).encode("utf-8"))
    if index != receipt.get("bundle_index_sha256"):
        raise BundleError("bundle index differs from receipt")
    return receipt


def write_bundle(
    output: Path,
    projected: dict[str, bytes],
    receipt: dict[str, Any],
    executable_paths: set[str],
) -> None:
    if output.exists() or output.is_symlink():
        raise BundleError(f"output already exists: {output}")
    output.mkdir(parents=True)
    for relative, payload in sorted(projected.items()):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        destination.chmod(0o755 if relative in executable_paths else 0o644)
    receipt_path = output / RECEIPT_NAME
    receipt_path.write_bytes(_json_bytes(receipt))
    receipt_path.chmod(0o644)


def write_deterministic_archive(bundle_root: Path, archive_path: Path, receipt: dict[str, Any]) -> str:
    if archive_path.exists() or archive_path.is_symlink():
        raise BundleError(f"archive already exists: {archive_path}")
    try:
        archive_path.resolve(strict=False).relative_to(bundle_root.resolve())
    except ValueError:
        pass
    else:
        raise BundleError("archive path must be outside the exported tree")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    file_map, _ = _receipt_maps(receipt)
    members = sorted(set(file_map) | {RECEIPT_NAME})
    root_name = receipt["archive_root"]
    executable_paths = set(receipt["executable_paths"])
    with archive_path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT) as archive:
                for relative in members:
                    payload = (bundle_root / relative).read_bytes()
                    info = tarfile.TarInfo(f"{root_name}/{relative}")
                    info.size = len(payload)
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mode = 0o755 if relative in executable_paths else 0o644
                    archive.addfile(info, io.BytesIO(payload))
    return _sha256(archive_path.read_bytes())


def _load_receipt_view(
    bundle_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    receipt = _load_json(bundle_root / RECEIPT_NAME, RECEIPT_NAME)
    files, omissions = _receipt_maps(receipt)
    unshipped = _unshipped_map(receipt, files, omissions)
    _verify_receipt_accounting(receipt, files, omissions, unshipped)
    return receipt, files, omissions, unshipped


def _original_sha_for(
    bundle_root: Path,
    relative: str,
    files: dict[str, dict[str, Any]],
    omissions: dict[str, dict[str, Any]],
    unshipped: dict[str, dict[str, Any]] | None = None,
) -> str:
    relative = _safe_relative(relative)
    if relative in files:
        record = files[relative]
        path = _assert_no_symlink_components(bundle_root, relative, "receipted source")
        payload = path.read_bytes()
        if len(payload) != record["export_bytes"] or _sha256(payload) != record["export_sha256"]:
            raise BundleError(f"receipted export bytes changed: {relative}")
        return record["original_sha256"]
    if relative in omissions:
        return omissions[relative]["original_sha256"]
    if unshipped is not None and relative in unshipped:
        return unshipped[relative]["original_sha256"]
    raise BundleError(
        f"path is outside receipt, omission, and unshipped inventories: {relative}"
    )


def _typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return list(left) == list(right) and all(
            _typed_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _typed_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _original_record_for(
    relative: str,
    files: dict[str, dict[str, Any]],
    omissions: dict[str, dict[str, Any]],
    unshipped: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    relative = _safe_relative(relative)
    for inventory in (files, omissions, unshipped):
        if relative in inventory:
            return inventory[relative]
    raise BundleError(f"path is outside every source-bundle receipt inventory: {relative}")


def _validate_procurl_registry_rows(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    diagnostics: dict[str, Any],
    unshipped: dict[str, dict[str, Any]],
) -> None:
    arms = manifest.get("schedule", {}).get("arms")
    seeds = manifest.get("schedule", {}).get("seeds")
    run_index = manifest.get("run_index")
    if (
        not isinstance(arms, list)
        or len(arms) != 4
        or not all(isinstance(arm, str) for arm in arms)
        or not isinstance(seeds, list)
        or seeds != list(range(21_000, 21_080))
        or not isinstance(run_index, list)
        or len(run_index) != 320
    ):
        raise BundleError("ProCuRL compact manifest schedule is malformed")
    development_seeds = [21_300, 21_301, 21_302]
    expected_coordinates = [
        ("development", arm, seed)
        for arm in arms
        for seed in development_seeds
    ] + [
        ("confirmatory", arm, seed)
        for arm in arms
        for seed in seeds
    ]
    observed_coordinates = [
        (row.get("mode"), row.get("arm"), row.get("seed")) for row in rows
    ]
    if not _typed_equal(observed_coordinates, expected_coordinates):
        raise BundleError("ProCuRL registry row order or coordinates changed")

    raw = unshipped.get(PROCURL_CONFIRMATORY_RAW)
    development = unshipped.get(PROCURL_DEVELOPMENT_RAW)
    if raw is None or development is None:
        raise BundleError("ProCuRL unshipped raw declarations are incomplete")
    for ordinal, row in enumerate(rows[:12]):
        arm_index, seed_index = divmod(ordinal, len(development_seeds))
        arm = arms[arm_index]
        expected = {
            "experiment": "acrobot_procurl_selection_semantics_development",
            "mode": "development",
            "arm": arm,
            "seed": development_seeds[seed_index],
            "status": "complete",
            "evidence_path": PROCURL_DEVELOPMENT_RAW,
            "raw_path": PROCURL_DEVELOPMENT_RAW,
            "raw_status": "vendored-aggregate-run-record",
            "source_lock_path": PROCURL_LOCK,
            "development_gate_path": PROCURL_GATE,
            "results_path": PROCURL_RESULTS,
        }
        if any(not _typed_equal(row.get(key), value) for key, value in expected.items()):
            raise BundleError(f"ProCuRL development registry row {ordinal} changed")
        if row.get("evidence_locator") != f"cases.{arm}.runs[{seed_index}]":
            raise BundleError(f"ProCuRL development locator {ordinal} changed")

    diagnostic_arms = diagnostics.get("arms")
    if not isinstance(diagnostic_arms, dict) or list(diagnostic_arms) != arms:
        raise BundleError("ProCuRL diagnostics arm order changed")
    for ordinal, row in enumerate(rows[12:]):
        arm_index, seed_index = divmod(ordinal, len(seeds))
        arm = arms[arm_index]
        seed = seeds[seed_index]
        index_record = run_index[ordinal]
        per_seed = diagnostic_arms[arm].get("per_seed")
        if (
            not isinstance(per_seed, list)
            or len(per_seed) != 80
            or type(per_seed[seed_index].get("seed")) is not int
            or per_seed[seed_index]["seed"] != seed
        ):
            raise BundleError(f"ProCuRL diagnostics row {ordinal} changed")
        expected_index = {
            "ordinal": ordinal,
            "arm": arm,
            "seed": seed,
        }
        if any(
            not _typed_equal(index_record.get(key), value)
            for key, value in expected_index.items()
        ):
            raise BundleError(f"ProCuRL manifest run index {ordinal} changed")
        expected = {
            "experiment": "acrobot_procurl_selection_semantics_confirmatory",
            "mode": "confirmatory",
            "arm": arm,
            "seed": seed,
            "status": "complete",
            "evidence_path": PROCURL_DIAGNOSTICS,
            "raw_path": None,
            "raw_status": "external-content-addressed-aggregate-run-record",
            "source_lock_path": PROCURL_LOCK,
            "development_gate_path": PROCURL_GATE,
            "derived_analysis_path": PROCURL_ANALYSIS,
            "portable_verification_path": PROCURL_PORTABLE,
            "external_raw_manifest_path": PROCURL_EXTERNAL_MANIFEST,
            "results_path": PROCURL_RESULTS,
            "raw_artifact_sha256": raw["original_sha256"],
            "raw_artifact_size_bytes": raw["original_bytes"],
            "raw_run_sha256": index_record.get("canonical_json_sha256"),
            "raw_run_size_bytes": index_record.get("canonical_json_size_bytes"),
            "content_addressed_download_uri": None,
        }
        if any(not _typed_equal(row.get(key), value) for key, value in expected.items()):
            raise BundleError(f"ProCuRL confirmatory registry row {ordinal} changed")
        if (
            row.get("raw_locator") != f"run_index[{ordinal}]"
            or row.get("evidence_locator")
            != f"arms.{arm}.per_seed[{seed_index}]"
        ):
            raise BundleError(f"ProCuRL confirmatory locator {ordinal} changed")


def check_run_registry(bundle_root: Path) -> None:
    _, files, omissions, unshipped = _load_receipt_view(bundle_root)
    module = importlib.import_module("curriculum_maxrl.build_run_registry")
    stored_path = bundle_root / "curriculum_maxrl/run_registry.json"
    stored_registry = _load_json(stored_path, "run registry")
    stored_rows = stored_registry.get("rows") if isinstance(stored_registry, dict) else None
    if not isinstance(stored_rows, list):
        raise BundleError("run registry rows are malformed")
    procurl_rows = [
        copy.deepcopy(row)
        for row in stored_rows
        if isinstance(row, dict)
        and str(row.get("experiment", "")).startswith(
            "acrobot_procurl_selection_semantics_"
        )
    ]
    manifest = _load_json(bundle_root / PROCURL_EXTERNAL_MANIFEST, "ProCuRL manifest")
    diagnostics = _load_json(bundle_root / PROCURL_DIAGNOSTICS, "ProCuRL diagnostics")
    _validate_procurl_registry_rows(procurl_rows, manifest, diagnostics, unshipped)
    original_sha = module._sha256
    original_procurl_rows = module._acrobot_procurl_selection_rows
    original_validate = module.validate_registry

    def receipt_sha(path: Path) -> str:
        relative = path.resolve().relative_to(bundle_root.resolve()).as_posix()
        return _original_sha_for(bundle_root, relative, files, omissions, unshipped)

    module._sha256 = receipt_sha
    module._acrobot_procurl_selection_rows = lambda: copy.deepcopy(procurl_rows)

    def receipt_validate(registry: dict[str, Any]) -> None:
        validation_copy = copy.deepcopy(registry)
        for row in validation_copy.get("rows", []):
            if row.get("experiment") == (
                "acrobot_procurl_selection_semantics_development"
            ):
                row["evidence_path"] = PROCURL_GATE
                row["raw_path"] = PROCURL_GATE
        original_validate(validation_copy)

    module.validate_registry = receipt_validate
    try:
        rendered = module._serialized(module.build_registry())
    finally:
        module._sha256 = original_sha
        module._acrobot_procurl_selection_rows = original_procurl_rows
        module.validate_registry = original_validate
    stored = stored_path.read_text(encoding="utf-8")
    if stored != rendered:
        raise BundleError("run registry differs under receipt-aware regeneration")


def check_acrobot_procurl_compact(bundle_root: Path) -> dict[str, int]:
    _, files, omissions, unshipped = _load_receipt_view(bundle_root)
    missing = sorted(set(PROCURL_SAFE_COMPACT_PATHS) - set(files))
    if missing:
        raise BundleError(f"ProCuRL compact source closure is incomplete: {missing}")
    expected_unshipped = {
        PROCURL_CONFIRMATORY_RAW: (PROCURL_RAW_SIZE, PROCURL_RAW_SHA256),
        PROCURL_DEVELOPMENT_RAW: (
            PROCURL_DEVELOPMENT_SIZE,
            PROCURL_DEVELOPMENT_SHA256,
        ),
    }
    if set(unshipped) != set(expected_unshipped):
        raise BundleError("ProCuRL unshipped artifact inventory differs")
    for relative, (expected_size, expected_sha) in expected_unshipped.items():
        record = unshipped[relative]
        if (
            type(record.get("original_bytes")) is not int
            or record["original_bytes"] != expected_size
            or record.get("original_sha256") != expected_sha
            or record.get("content_addressed_download_uri") is not None
        ):
            raise BundleError(f"ProCuRL unshipped declaration changed: {relative}")

    manifest = _load_json(bundle_root / PROCURL_EXTERNAL_MANIFEST, "ProCuRL manifest")
    validator = importlib.import_module(
        "frontier_rl.examples.build_acrobot_procurl_external_manifest"
    )
    try:
        validator.validate_manifest_shape(manifest)
    except (TypeError, ValueError) as exc:
        raise BundleError(f"ProCuRL compact manifest shape failed: {exc}") from exc

    expected_raw_binding = {
        "logical_path": PROCURL_CONFIRMATORY_RAW,
        "size_bytes": PROCURL_RAW_SIZE,
        "sha256": PROCURL_RAW_SHA256,
        "schema": "curriculum-maxrl/acrobot-procurl-selection-raw/v1",
    }
    if not _typed_equal(manifest.get("raw_artifact"), expected_raw_binding):
        raise BundleError("ProCuRL external raw binding differs")
    role_paths = {
        "source_lock": PROCURL_LOCK,
        "development_gate": PROCURL_GATE,
        "confirmatory_analysis": PROCURL_ANALYSIS,
        "portable_verification": PROCURL_PORTABLE,
    }
    bindings = manifest.get("bindings")
    if not isinstance(bindings, dict) or list(bindings) != list(role_paths):
        raise BundleError("ProCuRL manifest binding roles differ")
    for role, relative in role_paths.items():
        binding = bindings[role]
        source_record = _original_record_for(relative, files, omissions, unshipped)
        if (
            binding.get("logical_path") != relative
            or binding.get("sha256") != source_record["original_sha256"]
            or not _typed_equal(binding.get("size_bytes"), source_record["original_bytes"])
        ):
            raise BundleError(f"ProCuRL compact binding differs: {role}")

    lock = _load_json(bundle_root / PROCURL_LOCK, "ProCuRL source lock")
    source_manifest = lock.get("source_sha256") if isinstance(lock, dict) else None
    if not isinstance(source_manifest, dict) or not source_manifest:
        raise BundleError("ProCuRL source lock manifest is missing")
    omitted_locked: list[str] = []
    for relative, expected in source_manifest.items():
        if not isinstance(relative, str) or HEX64.fullmatch(str(expected)) is None:
            raise BundleError("ProCuRL source lock manifest is malformed")
        actual = _original_sha_for(
            bundle_root, relative, files, omissions, unshipped
        )
        if actual != expected:
            raise BundleError(f"ProCuRL locked original hash differs: {relative}")
        if relative in omissions:
            omitted_locked.append(relative)
    invalid_incident = (
        "frontier_rl/examples/INVALID_ABORTED_PRE_GATE_ENTROPY_SUM_MISMATCH/"
        "INCIDENT.json"
    )
    if omitted_locked != [invalid_incident]:
        raise BundleError("ProCuRL locked omission inventory differs")

    gate = _load_json(bundle_root / PROCURL_GATE, "ProCuRL development gate")
    analysis = _load_json(bundle_root / PROCURL_ANALYSIS, "ProCuRL analysis")
    portable = _load_json(bundle_root / PROCURL_PORTABLE, "ProCuRL portable receipt")
    diagnostics = _load_json(bundle_root / PROCURL_DIAGNOSTICS, "ProCuRL diagnostics")
    lock_sha = bindings["source_lock"]["sha256"]
    gate_sha = bindings["development_gate"]["sha256"]
    analysis_sha = bindings["confirmatory_analysis"]["sha256"]
    if (
        gate.get("schema")
        != "curriculum-maxrl/acrobot-procurl-selection-development-gates/v1"
        or gate.get("mode") != "development"
        or gate.get("all_gates_passed") is not True
        or not isinstance(gate.get("gates"), dict)
        or not all(value is True for value in gate["gates"].values())
        or gate.get("raw_artifact_relative_path") != PROCURL_DEVELOPMENT_RAW
        or gate.get("raw_artifact_sha256") != PROCURL_DEVELOPMENT_SHA256
        or gate.get("source_lock_sha256") != lock_sha
    ):
        raise BundleError("ProCuRL compact development-gate binding differs")
    development_binding = analysis.get("development_gate_binding_verification", {})
    primary = analysis.get("primary", {})
    if (
        analysis.get("schema")
        != "curriculum-maxrl/acrobot-procurl-selection-analysis/v1"
        or analysis.get("mode") != "confirmatory"
        or analysis.get("strict_validation_passed") is not True
        or analysis.get("raw_artifact_relative_path") != PROCURL_CONFIRMATORY_RAW
        or analysis.get("raw_artifact_sha256") != PROCURL_RAW_SHA256
        or analysis.get("source_lock_verification", {}).get("passed") is not True
        or analysis.get("source_lock_verification", {}).get("source_lock_sha256")
        != lock_sha
        or development_binding.get("passed") is not True
        or development_binding.get("development_gate_sha256") != gate_sha
        or development_binding.get("development_raw_relative_path")
        != PROCURL_DEVELOPMENT_RAW
        or development_binding.get("development_raw_sha256")
        != PROCURL_DEVELOPMENT_SHA256
        or primary.get("n_pairs") != 80
        or primary.get("supported") is not False
    ):
        raise BundleError("ProCuRL compact analysis binding differs")
    portable_source = portable.get("source_manifest_verification", {})
    portable_invalid = portable.get("invalid_pre_gate_archive_verification", {})
    if (
        portable.get("schema")
        != "curriculum-maxrl/acrobot-procurl-selection-portable-verification/v1"
        or portable.get("all_checks_passed") is not True
        or portable.get("source_lock_sha256") != lock_sha
        or portable_source.get("passed") is not True
        or not _typed_equal(
            portable_source.get("checked_source_files"), sorted(source_manifest)
        )
        or portable_invalid.get("passed") is not True
        or portable_invalid.get("incident_relative_path") != invalid_incident
        or portable_invalid.get("incident_sha256")
        != omissions[invalid_incident]["original_sha256"]
        or portable.get("raw_ledger_validation", {}).get("paired_seed_count") != 80
        or portable.get("raw_ledger_validation", {}).get("arm_count") != 4
        or portable.get("stored_analysis_comparison", {}).get(
            "stored_analysis_sha256"
        )
        != analysis_sha
    ):
        raise BundleError("ProCuRL compact portable receipt binding differs")
    if (
        diagnostics.get("schema")
        != "curriculum-maxrl/acrobot-procurl-selection-descriptive-diagnostics/v1"
        or diagnostics.get("mode") != "confirmatory"
        or diagnostics.get("status") != "descriptive_only_no_new_inference"
        or not _typed_equal(diagnostics.get("raw_artifact"), expected_raw_binding)
        or not _typed_equal(diagnostics.get("source_lock"), bindings["source_lock"])
        or not _typed_equal(
            diagnostics.get("development_gate"), bindings["development_gate"]
        )
        or diagnostics.get("schedule", {}).get("run_count") != 320
        or diagnostics.get("metric_policy", {}).get("new_inferential_statistics")
        is not False
    ):
        raise BundleError("ProCuRL compact diagnostics binding differs")

    stored_registry = _load_json(
        bundle_root / "curriculum_maxrl/run_registry.json", "run registry"
    )
    registry_rows = [
        row
        for row in stored_registry.get("rows", [])
        if isinstance(row, dict)
        and str(row.get("experiment", "")).startswith(
            "acrobot_procurl_selection_semantics_"
        )
    ]
    _validate_procurl_registry_rows(
        registry_rows, manifest, diagnostics, unshipped
    )
    return {
        "selected_compact_files": len(PROCURL_SAFE_COMPACT_PATHS),
        "locked_source_files": len(source_manifest),
        "receipted_locked_omissions": len(omitted_locked),
        "registry_rows": len(registry_rows),
        "unshipped_bytes": sum(
            record["original_bytes"] for record in unshipped.values()
        ),
    }


def check_v3_audit(bundle_root: Path) -> None:
    _, files, omissions, _ = _load_receipt_view(bundle_root)
    module = importlib.import_module("frontier_rl.examples.analyze_acrobot_v3_mechanism")
    artifact_path = bundle_root / "frontier_rl/examples/acrobot_neural_v3_shared_confirmatory.json"
    lock_path = bundle_root / "frontier_rl/examples/ACROBOT_NEURAL_V3_LOCK.json"
    stored_path = bundle_root / "frontier_rl/examples/acrobot_v3_mechanism_audit.json"
    artifact = _load_json(artifact_path)
    lock = _load_json(lock_path)
    result = module.analyze(artifact, lock, artifact_path, lock_path)
    result["input"]["source_lock_sha256"] = _original_sha_for(
        bundle_root,
        "frontier_rl/examples/ACROBOT_NEURAL_V3_LOCK.json",
        files,
        omissions,
    )
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if stored_path.read_text(encoding="utf-8") != rendered:
        raise BundleError("Acrobot V3 mechanism audit differs under original-hash binding")


DIGITS_EXPECTED = {
    "SOURCE_LOCK.json": "d72b93a29a2e6a096a6acb0611f69fe6df9dcda80256000aed6de5208ef4eb36",
    "digits_split_manifest.json": "13dbc30cc5143edb043d76d76aac18bcc3a456b174a18ba488498fb99eab5e3f",
    "engineering/reseal_v3/engineering_audit.json": "d448a22793dc5a52ae7809dfc7572e3d669bebc5f8ef8d00a9865a0bb16150d1",
    "engineering/reseal_v3/independent_preseal_review.json": "0c387ddd1b2bb49d2be6e1eacff96c9281cfa17eea85f01b425733c9aabe24ff",
    "authorizations/development_authorization.json": "91ec2c5dbea8e4d424abcbb8176df9bfc8e809f46248b8345d41ea530fbbff61",
    "analyses/development_registered_v1/lr_selection.json": "dfc9d69faec78cff95e63ed7cd99a0e23c883dad5eba1a2fe378366730e06795",
    "analyses/development_registered_v1/independent_preconfirmation_review.json": "6ad8ede4ebdc6517e7d44395ae14dc26cc705029c1efab454eabb366d729803a",
    "authorizations/confirmation_tuned_authorization.json": "4d8967d9cb9a499c9cb3f439385d155e3df65fae7c7aed94260b81d1501f71ca",
    "authorizations/confirmation_common_authorization.json": "8e09bf3f43f5f6ede5f1b623dfd2d89b7488a7171a87aa9818da53f1f4ec13ab",
    "analyses/confirmation_registered_v1/confirmation_analysis.json": "346e46414d82155f2064ee2a448b89cf976bdc6897e0ff8ced06a389056799d6",
    "analyses/confirmation_registered_v1/common_identity_robustness.json": "01b2e07de7eaef8e66da938285df90cb9d871b77ffaa9195cfba1fc7ca14b85e",
    "analyses/confirmation_registered_v1/confirmation_bundle_receipt.json": "002eba15698ddc91e7360c8f3795cd1dc9562595338658c13bcdcc315b9fb3d6",
}


def check_digits_chain(bundle_root: Path) -> None:
    _, files, omissions, _ = _load_receipt_view(bundle_root)
    prefix = "curriculum_maxrl/digits_factorial/"
    for relative, expected in DIGITS_EXPECTED.items():
        actual = _original_sha_for(bundle_root, prefix + relative, files, omissions)
        if actual != expected:
            raise BundleError(f"Digits original hash differs: {relative}: {actual} != {expected}")
    root = bundle_root / "curriculum_maxrl/digits_factorial"
    load = lambda relative: _load_json(root / relative, relative)
    selection = load("analyses/development_registered_v1/lr_selection.json")
    analysis = load("analyses/confirmation_registered_v1/confirmation_analysis.json")
    identity = load("analyses/confirmation_registered_v1/common_identity_robustness.json")
    receipt = load("analyses/confirmation_registered_v1/confirmation_bundle_receipt.json")
    development = load("authorizations/development_authorization.json")
    tuned = load("authorizations/confirmation_tuned_authorization.json")
    common = load("authorizations/confirmation_common_authorization.json")
    expected = DIGITS_EXPECTED
    checks = [
        selection["all_development_gates_passed"] is True,
        selection["selected_learning_rates_by_estimator"] == {"practical_maxrl": 0.1, "rloo": 0.1},
        selection["selected_common_learning_rate"] == 0.1,
        selection["development_authorization"]["sha256"] == expected["authorizations/development_authorization.json"],
        development["source_lock_sha256"] == expected["SOURCE_LOCK.json"],
        development["zero_lr_engineering_audit"]["sha256"] == expected["engineering/reseal_v3/engineering_audit.json"],
        analysis["lr_selection_sha256"] == expected["analyses/development_registered_v1/lr_selection.json"],
        analysis["source_lock_sha256"] == expected["SOURCE_LOCK.json"],
        analysis["tuned"]["n_complete_blocks"] == 24,
        analysis["tuned"]["cell_failures"] == [],
        analysis["tuned"]["treatment_delivery"]["passed"] is True,
        analysis["tuned"]["primary_supported"] is False,
        analysis["tuned"]["contrasts"]["interaction"]["exact_two_sided_sign_flip_p"] == 0.34955739974975586,
        identity["result"]["all_ledgers_and_five_recovery_checkpoints_byte_identical"] is True,
        identity["result"]["optimizer_sensitivity_variation_present"] is False,
        identity["paired_run_count"] == 144,
        identity["paired_binary_file_count"] == 864,
        receipt["source_lock_sha256"] == expected["SOURCE_LOCK.json"],
        receipt["learning_rate_selection_sha256"] == expected["analyses/development_registered_v1/lr_selection.json"],
        receipt["preconfirmation_review_sha256"] == expected["analyses/development_registered_v1/independent_preconfirmation_review.json"],
        receipt["confirmation_analysis_sha256"] == expected["analyses/confirmation_registered_v1/confirmation_analysis.json"],
        receipt["common_identity_robustness_sha256"] == expected["analyses/confirmation_registered_v1/common_identity_robustness.json"],
        receipt["tuned_schedule"]["authorization_sha256"] == expected["authorizations/confirmation_tuned_authorization.json"],
        receipt["common_schedule"]["authorization_sha256"] == expected["authorizations/confirmation_common_authorization.json"],
    ]
    for authorization, phase in ((tuned, "confirmation_tuned"), (common, "confirmation_common")):
        checks.extend(
            [
                authorization["authorized_phase"] == phase,
                authorization["source_lock_sha256"] == expected["SOURCE_LOCK.json"],
                authorization["lr_selection"]["sha256"] == expected["analyses/development_registered_v1/lr_selection.json"],
                authorization["independent_preseal_review"]["review_sha256"] == expected["analyses/development_registered_v1/independent_preconfirmation_review.json"],
            ]
        )
    for schedule in ("tuned_schedule", "common_schedule"):
        checks.extend(
            [
                receipt[schedule]["run_count"] == 144,
                receipt[schedule]["complete_blocks"] == 24,
                receipt[schedule]["failures"] == 0,
                receipt[schedule]["paid_actions"] == 37_748_736,
            ]
        )
    if not all(checks):
        raise BundleError("Digits receipt-aware scientific chain check failed")


def check_paper_manifest(bundle_root: Path) -> int:
    _, files, omissions, unshipped = _load_receipt_view(bundle_root)
    manifest = _load_json(bundle_root / "paper/results/manifest.json", "paper manifest")
    checks: list[tuple[str, str]] = []
    for section in ("figures", "results"):
        for entry in manifest.get(section, {}).values():
            checks.extend(entry.get("checksums", {}).items())
    timing = manifest.get("timing", {})
    if timing.get("artifact") and timing.get("checksum"):
        checks.append((timing["artifact"], timing["checksum"]))
    for relative, expected in checks:
        actual = _original_sha_for(
            bundle_root, relative, files, omissions, unshipped
        )[: len(expected)]
        if actual != expected:
            raise BundleError(f"paper manifest mismatch: {relative}: {actual} != {expected}")
    return len(checks)


@contextmanager
def _patched_portable_hashes(bundle_root: Path):
    _, files, omissions, _ = _load_receipt_view(bundle_root)
    portable = importlib.import_module(
        "frontier_rl.examples.verify_acrobot_curriculum_tournament_portable"
    )
    analysis = importlib.import_module(
        "frontier_rl.examples.analyze_acrobot_curriculum_tournament"
    )
    original_sha = portable._sha256
    original_manifest = portable._verify_source_manifest

    def receipt_sha(path: Path) -> str:
        relative = path.resolve().relative_to(bundle_root.resolve()).as_posix()
        return _original_sha_for(bundle_root, relative, files, omissions)

    def receipt_manifest(lock: dict[str, Any], source_root: Path) -> dict[str, Any]:
        manifest = lock.get("source_sha256")
        expected_paths = set(analysis.EXPECTED_SOURCE_RELATIVE_PATHS)
        if not isinstance(manifest, dict) or set(manifest) != expected_paths:
            raise ValueError("source lock does not contain the exact frozen source manifest")
        checked: list[str] = []
        omitted: list[str] = []
        for relative, expected in manifest.items():
            actual = _original_sha_for(bundle_root, relative, files, omissions)
            if actual != expected:
                raise ValueError(f"locked original source hash mismatch: {relative}")
            checked.append(relative)
            if relative in omissions:
                omitted.append(relative)
        analyzer_relative = portable.LOCKED_ANALYZER_RELATIVE_PATH
        imported = Path(analysis.__file__).resolve().relative_to(bundle_root.resolve()).as_posix()
        if imported != analyzer_relative or receipt_sha(Path(analysis.__file__)) != manifest[analyzer_relative]:
            raise ValueError("imported analyzer bytes differ from the receipted original")
        return {
            "passed": True,
            "source_root_checked": True,
            "checked_source_files": sorted(checked),
            "exact_manifest_key_set": True,
            "all_live_hashes_match": True,
            "imported_analyzer_hash_matches_lock": True,
            "omitted_historical_witnesses_verified_by_receipt": sorted(omitted),
        }

    portable._sha256 = receipt_sha
    portable._verify_source_manifest = receipt_manifest
    try:
        yield portable
    finally:
        portable._sha256 = original_sha
        portable._verify_source_manifest = original_manifest


def check_acrobot_tournament(bundle_root: Path) -> None:
    with _patched_portable_hashes(bundle_root) as portable:
        result = portable.verify_portable(
            bundle_root / "frontier_rl/examples/ACROBOT_CURRICULUM_TOURNAMENT_LOCK.json",
            bundle_root / "frontier_rl/examples/acrobot_curriculum_tournament_confirmatory.json",
            bundle_root / "frontier_rl/examples/acrobot_curriculum_tournament_analysis.json",
            source_root=bundle_root,
        )
    if result.get("all_checks_passed") is not True:
        raise BundleError("Acrobot tournament receipt-aware portable verification failed")
    omitted = result["source_manifest_verification"].get(
        "omitted_historical_witnesses_verified_by_receipt", []
    )
    if len(omitted) != 2:
        raise BundleError("Acrobot tournament omission receipt count differs")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--scope", type=Path, default=Path(DEFAULT_SCOPE))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--archive", type=Path, default=Path(DEFAULT_ARCHIVE))
    parser.add_argument("--bundle-root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-source", action="store_true")
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify-export", action="store_true")
    mode.add_argument("--check-run-registry", action="store_true")
    mode.add_argument("--check-v3-audit", action="store_true")
    mode.add_argument("--check-digits-chain", action="store_true")
    mode.add_argument("--check-paper-manifest", action="store_true")
    mode.add_argument("--check-acrobot-tournament", action="store_true")
    mode.add_argument("--check-acrobot-procurl-compact", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.audit_source or args.build:
            repo_root = args.repo_root.absolute()
            scope_path = args.scope if args.scope.is_absolute() else repo_root / args.scope
            projected, receipt, executables = build_bundle(repo_root, scope_path)
            if args.audit_source:
                action = "audited"
                extra = ""
            else:
                output = args.output if args.output.is_absolute() else repo_root / args.output
                archive = args.archive if args.archive.is_absolute() else repo_root / args.archive
                write_bundle(output, projected, receipt, executables)
                verify_export(output)
                archive_sha = write_deterministic_archive(output, archive, receipt)
                action = "built"
                extra = f", archive_sha256={archive_sha}"
            summary = receipt["summary"]
            print(
                f"Anonymous source bundle {action}: "
                f"{summary['selected_file_count']} selected + 1 receipt, "
                f"groups={summary['group_file_counts']}, "
                f"{summary['transformed_file_count']} transformed files, "
                f"{summary['omitted_witness_count']} omission receipts, "
                f"{summary['unshipped_artifact_count']} unshipped declarations "
                f"({summary['unshipped_artifact_bytes']} bytes, 0 download URIs), "
                f"0 leaks{extra}"
            )
            return 0

        bundle_root = args.bundle_root.absolute()
        if args.verify_export:
            receipt = verify_export(bundle_root)
            print(
                f"Anonymous source bundle verified: "
                f"{receipt['summary']['selected_file_count']} selected + 1 receipt, 0 leaks"
            )
        elif args.check_run_registry:
            check_run_registry(bundle_root)
            print("Run registry matches under original-hash receipt binding")
        elif args.check_v3_audit:
            check_v3_audit(bundle_root)
            print("Acrobot V3 audit matches under original-hash receipt binding")
        elif args.check_digits_chain:
            check_digits_chain(bundle_root)
            print("Digits locked receipt chain matches original hashes")
        elif args.check_paper_manifest:
            checked = check_paper_manifest(bundle_root)
            print(f"Paper manifest original hashes pass: {checked} inputs checked, 0 mismatches")
        elif args.check_acrobot_tournament:
            check_acrobot_tournament(bundle_root)
            print("Acrobot tournament passes with two receipted historical omissions")
        elif args.check_acrobot_procurl_compact:
            result = check_acrobot_procurl_compact(bundle_root)
            print(
                "Acrobot ProCuRL compact chain passes: "
                f"{result['selected_compact_files']} safe compact files, "
                f"{result['locked_source_files']} locked source hashes, "
                f"{result['receipted_locked_omissions']} locked omission, "
                f"{result['registry_rows']} registry rows; "
                f"{result['unshipped_bytes']} raw bytes absent, 0 download URIs; "
                "full raw replay was not run"
            )
        return 0
    except (BundleError, ValueError, OSError) as exc:
        print(f"anonymous source bundle failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
