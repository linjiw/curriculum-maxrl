"""Build a deterministic, non-mutating anonymous evidence release.

The source evidence remains byte-for-byte untouched. Only explicitly approved
JSON leaves and count-locked exact text matches may change in the projected
copy. Every other forbidden path or identity is a hard error.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


SCOPE_SCHEMA = "curriculum-maxrl/anonymous-release-scope/v2"
SCHEMA = "curriculum-maxrl/anonymous-release-receipt/v2"
COMPACT_SCHEMA = "curriculum-maxrl/anonymous-release-compact-receipt/v2"
DEFAULT_SCOPE = "anonymous_release_scope.json"
DEFAULT_OUTPUT = "tmp/anonymous-release"
RECEIPT_NAME = "ANONYMIZATION_RECEIPT.json"
ALLOWED_SUFFIXES = {".bib", ".html", ".json", ".md", ".py", ".sh", ".tex", ".txt"}
WINDOWS_USER_PATH = re.compile(r"(?i)(?:^|[^A-Za-z0-9])[A-Z]:\\Users\\")
WINDOWS_UNC_PATH = re.compile(r"(?i)(?:^|[^\\])\\\\[^\\\s]+\\[^\\\s]+")
IDENTITY_TOKEN = re.compile(r"[A-Za-z0-9_-]+")
GITHUB_OWNER_URL = re.compile(r"https://github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)")
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


class ProjectionError(RuntimeError):
    """The requested projection is incomplete, unsafe, or stale."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ProjectionError(f"scope path is not a safe repository-relative path: {value!r}")
    return path.as_posix()


def _assert_not_never_read(relative: str) -> None:
    folded = relative.casefold()
    for pattern in NEVER_READ_GLOBS:
        if fnmatch.fnmatchcase(folded, pattern.casefold()):
            raise ProjectionError(
                "path is forbidden before any read by the external-raw policy: "
                f"{relative} ({pattern})"
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
                raise ProjectionError(f"non-string JSON key at {pointer or '/'}")
            yield from _walk_json_strings(child, f"{pointer}/{_pointer_part(key)}")


@dataclass(frozen=True)
class ForbiddenPolicy:
    markers: tuple[str, ...]
    case_insensitive_markers: tuple[str, ...]
    identity_sha256: frozenset[str]

    def contains(self, value: str) -> bool:
        folded = value.casefold()
        return (
            any(marker in value for marker in self.markers)
            or any(marker.casefold() in folded for marker in self.case_insensitive_markers)
            or any(
                _sha256(token.group(0).casefold().encode("utf-8")) in self.identity_sha256
                for token in IDENTITY_TOKEN.finditer(value)
            )
            or WINDOWS_USER_PATH.search(value) is not None
            or WINDOWS_UNC_PATH.search(value) is not None
        )


def _assert_json_keys_clean(
    value: Any, policy: ForbiddenPolicy, pointer: str = ""
) -> None:
    if isinstance(value, list):
        for index, child in enumerate(value):
            _assert_json_keys_clean(child, policy, f"{pointer}/{index}")
    elif isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ProjectionError(f"non-string JSON key at {pointer or '/'}")
            if policy.contains(key):
                raise ProjectionError(
                    f"local absolute path in JSON key at {pointer or '/'}; keys cannot be projected"
                )
            _assert_json_keys_clean(child, policy, f"{pointer}/{_pointer_part(key)}")


def _json_pointer_set(value: Any, pointer: str, replacement: str) -> None:
    if not pointer.startswith("/"):
        raise ProjectionError(f"only non-root JSON pointers are supported: {pointer!r}")
    parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]
    parent = value
    for part in parts[:-1]:
        if isinstance(parent, list):
            try:
                parent = parent[int(part)]
            except (ValueError, IndexError) as exc:
                raise ProjectionError(f"invalid JSON pointer: {pointer}") from exc
        elif isinstance(parent, dict) and part in parent:
            parent = parent[part]
        else:
            raise ProjectionError(f"invalid JSON pointer: {pointer}")
    leaf = parts[-1]
    if isinstance(parent, list):
        try:
            index = int(leaf)
            old = parent[index]
        except (ValueError, IndexError) as exc:
            raise ProjectionError(f"invalid JSON pointer: {pointer}") from exc
        if not isinstance(old, str):
            raise ProjectionError(f"approved JSON pointer is not a string: {pointer}")
        parent[index] = replacement
    elif isinstance(parent, dict) and leaf in parent:
        if not isinstance(parent[leaf], str):
            raise ProjectionError(f"approved JSON pointer is not a string: {pointer}")
        parent[leaf] = replacement
    else:
        raise ProjectionError(f"invalid JSON pointer: {pointer}")


def _policy(scope: dict[str, Any]) -> ForbiddenPolicy:
    return ForbiddenPolicy(
        markers=tuple(scope["forbidden_prefixes"]),
        case_insensitive_markers=tuple(scope["forbidden_literals"]),
        identity_sha256=frozenset(scope["forbidden_identity_sha256"]),
    )


def _assert_scope_metadata_clean(scope: dict[str, Any], policy: ForbiddenPolicy) -> None:
    """Scan scope metadata while exempting only explicit detector/match declarations."""

    metadata = copy.deepcopy(scope)
    metadata.pop("forbidden_prefixes")
    metadata.pop("forbidden_literals")
    metadata.pop("forbidden_identity_sha256")
    for transform in metadata["approved_text_transforms"]:
        if "match" in transform:
            transform["match"] = "<DECLARED_FORBIDDEN_MATCH>"
    rendered = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    if policy.contains(rendered):
        raise ProjectionError("forbidden content in non-detector scope metadata")


def _assert_regular_inside(repo_root: Path, relative: str) -> Path:
    """Reject file/ancestor symlinks and any resolution outside ``repo_root``."""

    _assert_not_never_read(relative)
    candidate = repo_root
    for part in PurePosixPath(relative).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ProjectionError(f"selected path has a symlink component: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ProjectionError(f"selected file is missing or unreadable: {relative}") from exc
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ProjectionError(f"selected path resolves outside repository: {relative}") from exc
    if not resolved.is_file():
        raise ProjectionError(f"selected path is not a regular file: {relative}")
    return resolved


def _assert_no_symlink_components(root: Path, relative: str, label: str) -> None:
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ProjectionError(f"{label} has a symlink component: {relative}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProjectionError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except ProjectionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectionError(f"cannot parse JSON file {label}: {exc}") from exc


def _load_json(path: Path) -> Any:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProjectionError(f"cannot read JSON file {path}: {exc}") from exc
    return _parse_json_bytes(payload, str(path))


def load_scope(repo_root: Path, scope_path: Path) -> dict[str, Any]:
    scope = _load_json(scope_path)
    if not isinstance(scope, dict):
        raise ProjectionError("anonymous-release scope must be a JSON object")
    if scope.get("schema") != SCOPE_SCHEMA:
        raise ProjectionError("unsupported anonymous-release scope schema")
    allowed_fields = {
        "schema",
        "paper_manifest",
        "entrypoints",
        "manifest_expansion",
        "unshipped_artifacts",
        "manifest_input_exclusions",
        "forbidden_prefixes",
        "forbidden_literals",
        "forbidden_identity_sha256",
        "approved_json_transforms",
        "approved_text_transforms",
        "excluded_globs",
    }
    unexpected = set(scope) - allowed_fields
    if unexpected:
        raise ProjectionError(f"unknown scope field(s): {sorted(unexpected)}")
    required_lists = (
        "entrypoints",
        "unshipped_artifacts",
        "manifest_input_exclusions",
        "forbidden_prefixes",
        "forbidden_literals",
        "forbidden_identity_sha256",
        "approved_json_transforms",
        "approved_text_transforms",
        "excluded_globs",
    )
    for key in required_lists:
        if not isinstance(scope.get(key), list):
            raise ProjectionError(f"scope field {key!r} must be a list")
    if not scope["forbidden_prefixes"]:
        raise ProjectionError("scope must declare at least one forbidden prefix")
    for prefix in scope["forbidden_prefixes"]:
        if not isinstance(prefix, str) or not prefix.startswith("/"):
            raise ProjectionError(f"invalid forbidden prefix: {prefix!r}")
    for field in ("forbidden_literals",):
        for value in scope[field]:
            if not isinstance(value, str) or not value:
                raise ProjectionError(f"invalid {field} entry: {value!r}")
    for digest in scope["forbidden_identity_sha256"]:
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ProjectionError(f"invalid forbidden identity digest: {digest!r}")
    _safe_relative(scope["paper_manifest"])
    for value in scope["entrypoints"]:
        _safe_relative(value)
    unshipped_paths: set[str] = set()
    previous = ""
    unshipped_fields = {
        "path",
        "original_bytes",
        "original_sha256",
        "availability",
        "content_addressed_download_uri",
        "reason",
    }
    for record in scope["unshipped_artifacts"]:
        if not isinstance(record, dict) or set(record) != unshipped_fields:
            raise ProjectionError("unshipped artifact declaration is malformed")
        relative = _safe_relative(record["path"])
        if relative <= previous or relative in unshipped_paths:
            raise ProjectionError("unshipped artifact inventory must be sorted and unique")
        previous = relative
        if type(record["original_bytes"]) is not int or record["original_bytes"] <= 0:
            raise ProjectionError(f"invalid unshipped byte count: {relative}")
        if (
            not isinstance(record["original_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", record["original_sha256"]) is None
        ):
            raise ProjectionError(f"invalid unshipped SHA-256: {relative}")
        if not isinstance(record["availability"], str) or not record["availability"]:
            raise ProjectionError(f"missing unshipped availability: {relative}")
        if record["content_addressed_download_uri"] is not None:
            raise ProjectionError(f"unshipped artifact advertises a download URI: {relative}")
        if not isinstance(record["reason"], str) or not record["reason"]:
            raise ProjectionError(f"missing unshipped reason: {relative}")
        unshipped_paths.add(relative)
    exclusion_paths: set[str] = set()
    previous = ""
    for record in scope["manifest_input_exclusions"]:
        if not isinstance(record, dict) or set(record) != {"path", "reason"}:
            raise ProjectionError("manifest-input exclusion is malformed")
        relative = _safe_relative(record["path"])
        if relative <= previous or relative in exclusion_paths:
            raise ProjectionError(
                "manifest-input exclusion inventory must be sorted and unique"
            )
        previous = relative
        if relative not in unshipped_paths:
            raise ProjectionError(
                f"manifest-input exclusion lacks an unshipped declaration: {relative}"
            )
        if not isinstance(record["reason"], str) or not record["reason"]:
            raise ProjectionError(f"missing manifest-input exclusion reason: {relative}")
        exclusion_paths.add(relative)
    for relative in unshipped_paths:
        if _is_excluded(relative, scope) is None:
            raise ProjectionError(
                f"unshipped artifact is outside the exclusion policy: {relative}"
            )
    return scope


def _is_excluded(relative: str, scope: dict[str, Any]) -> str | None:
    for item in scope["excluded_globs"]:
        pattern = item.get("glob")
        reason = item.get("reason")
        if not isinstance(pattern, str) or not isinstance(reason, str) or not reason:
            raise ProjectionError("each excluded_globs item needs non-empty glob and reason")
        if fnmatch.fnmatchcase(relative, pattern):
            return reason
    return None


def _verify_manifest_checksums(
    repo_root: Path,
    manifest: dict[str, Any],
    unshipped: dict[str, dict[str, Any]],
    manifest_exclusions: set[str],
) -> set[str]:
    checks: list[tuple[str, str]] = []
    timing = manifest.get("timing", {})
    if timing.get("artifact") and timing.get("checksum"):
        checks.append((timing["artifact"], timing["checksum"]))
    for section in ("results", "figures"):
        for entry in manifest.get(section, {}).values():
            checks.extend(entry.get("checksums", {}).items())
    observed_exclusions: set[str] = set()
    for raw_relative, expected in checks:
        relative = _safe_relative(raw_relative)
        if relative in manifest_exclusions:
            record = unshipped[relative]
            if record["original_sha256"][: len(expected)] != expected:
                raise ProjectionError(
                    f"manifest/declaration checksum mismatch for {relative}"
                )
            observed_exclusions.add(relative)
            continue
        _assert_not_never_read(relative)
        path = _assert_regular_inside(repo_root, relative)
        actual = _sha256(path.read_bytes())[: len(expected)]
        if actual != expected:
            raise ProjectionError(
                f"manifest checksum mismatch for {relative}: {actual} != {expected}"
            )
    return observed_exclusions


def resolve_scope(repo_root: Path, scope: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    manifest_relative = _safe_relative(scope["paper_manifest"])
    manifest_path = _assert_regular_inside(repo_root, manifest_relative)
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ProjectionError("paper manifest must be a JSON object")
    unshipped = {record["path"]: record for record in scope["unshipped_artifacts"]}
    manifest_exclusions = {
        record["path"] for record in scope["manifest_input_exclusions"]
    }
    observed_exclusions = _verify_manifest_checksums(
        repo_root, manifest, unshipped, manifest_exclusions
    )
    selected = {manifest_relative, *(_safe_relative(value) for value in scope["entrypoints"])}
    expansion = scope.get("manifest_expansion", {})
    manuscript = manifest.get("manuscript", {})
    if expansion.get("include_manuscript_sources"):
        for key in ("body", "extended_record"):
            if manuscript.get(key):
                selected.add(_safe_relative(manuscript[key]))
        for wrapper in manuscript.get("wrappers", []):
            selected.add(_safe_relative(wrapper))
    if expansion.get("include_timing_artifact") and manifest.get("timing", {}).get("artifact"):
        selected.add(_safe_relative(manifest["timing"]["artifact"]))
    for section, include_inputs in (
        ("results", expansion.get("include_result_inputs")),
        ("figures", expansion.get("include_figure_inputs")),
    ):
        for entry in manifest.get(section, {}).values():
            if include_inputs:
                for value in entry.get("inputs", []):
                    relative = _safe_relative(value)
                    if relative in manifest_exclusions:
                        observed_exclusions.add(relative)
                    else:
                        _assert_not_never_read(relative)
                        selected.add(relative)
            if section == "figures" and expansion.get("include_figure_scripts"):
                if entry.get("script"):
                    selected.add(_safe_relative(entry["script"]))
    for relative in sorted(selected):
        _assert_not_never_read(relative)
        reason = _is_excluded(relative, scope)
        if reason:
            raise ProjectionError(f"selected file is covered by an exclusion: {relative}: {reason}")
        path = _assert_regular_inside(repo_root, relative)
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ProjectionError(f"selected file is not an approved text/JSON type: {relative}")
    if observed_exclusions != manifest_exclusions:
        raise ProjectionError(
            "manifest-input exclusion inventory is stale: "
            f"missing={sorted(manifest_exclusions - observed_exclusions)}, "
            f"unexpected={sorted(observed_exclusions - manifest_exclusions)}"
        )
    return sorted(selected), manifest


@dataclass(frozen=True)
class Approval:
    file: str
    json_pointer: str
    replacement: str


@dataclass(frozen=True)
class TextApproval:
    file: str
    match: str | None
    github_owner_sha256: str | None
    replacement: str
    expected_occurrences: int
    purpose: str


def _json_approvals(
    scope: dict[str, Any], selected: set[str], policy: ForbiddenPolicy
) -> dict[tuple[str, str], Approval]:
    result: dict[tuple[str, str], Approval] = {}
    for raw in scope["approved_json_transforms"]:
        approval = Approval(
            file=_safe_relative(raw.get("file", "")),
            json_pointer=raw.get("json_pointer", ""),
            replacement=raw.get("replacement", ""),
        )
        if approval.file not in selected:
            raise ProjectionError(f"transform approval targets an out-of-scope file: {approval.file}")
        if not approval.file.endswith(".json") or not approval.json_pointer.startswith("/"):
            raise ProjectionError(f"invalid transform approval: {approval}")
        if not approval.replacement or policy.contains(approval.replacement):
            raise ProjectionError(f"unsafe transform replacement: {approval}")
        key = (approval.file, approval.json_pointer)
        if key in result:
            raise ProjectionError(f"duplicate transform approval: {key}")
        result[key] = approval
    return result


def _text_approvals(
    scope: dict[str, Any], selected: set[str], policy: ForbiddenPolicy
) -> dict[str, list[TextApproval]]:
    result: dict[str, list[TextApproval]] = {}
    seen: set[tuple[str, str]] = set()
    for raw in scope["approved_text_transforms"]:
        approval = TextApproval(
            file=_safe_relative(raw.get("file", "")),
            match=raw.get("match"),
            github_owner_sha256=raw.get("github_owner_sha256"),
            replacement=raw.get("replacement", ""),
            expected_occurrences=raw.get("expected_occurrences", 0),
            purpose=raw.get("purpose", "anonymization"),
        )
        if approval.file not in selected:
            raise ProjectionError(f"text approval targets an out-of-scope file: {approval.file}")
        if approval.file.endswith(".json"):
            raise ProjectionError(f"JSON files require pointer transforms: {approval.file}")
        if bool(approval.match) == bool(approval.github_owner_sha256):
            raise ProjectionError(
                f"text approval needs exactly one match or github-owner digest: {approval}"
            )
        if approval.purpose not in {"anonymization", "evidence_only_scope"}:
            raise ProjectionError(f"invalid text transform purpose: {approval}")
        if approval.match and approval.purpose == "anonymization" and not policy.contains(
            approval.match
        ):
            raise ProjectionError(f"text approval does not target forbidden content: {approval}")
        if approval.github_owner_sha256:
            if re.fullmatch(r"[0-9a-f]{64}", approval.github_owner_sha256) is None:
                raise ProjectionError(f"invalid GitHub-owner digest: {approval}")
            if approval.github_owner_sha256 not in policy.identity_sha256:
                raise ProjectionError(f"unregistered GitHub-owner digest: {approval}")
        if not approval.replacement or policy.contains(approval.replacement):
            raise ProjectionError(f"unsafe text replacement: {approval}")
        if not isinstance(approval.expected_occurrences, int) or approval.expected_occurrences < 1:
            raise ProjectionError(f"invalid text occurrence count: {approval}")
        key = (
            approval.file,
            approval.match or f"github-owner-sha256:{approval.github_owner_sha256}",
        )
        if key in seen:
            raise ProjectionError(f"duplicate text transform approval: {key}")
        seen.add(key)
        result.setdefault(approval.file, []).append(approval)
    return result


def _project_json(
    relative: str,
    source: bytes,
    policy: ForbiddenPolicy,
    approvals: dict[tuple[str, str], Approval],
) -> tuple[bytes, list[dict[str, Any]], set[tuple[str, str]]]:
    original = _parse_json_bytes(source, relative)
    _assert_json_keys_clean(original, policy)
    projected = copy.deepcopy(original)
    transformations: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    for pointer, value in _walk_json_strings(original):
        if not policy.contains(value):
            continue
        key = (relative, pointer)
        approval = approvals.get(key)
        if approval is None:
            raise ProjectionError(f"unapproved local path at {relative}#{pointer or '/'}")
        _json_pointer_set(projected, pointer, approval.replacement)
        used.add(key)
        transformations.append(
            {
                "kind": "json_pointer",
                "json_pointer": pointer,
                "original_value_sha256": _sha256(value.encode("utf-8")),
                "replacement": approval.replacement,
                "occurrence_count": 1,
            }
        )
    for pointer, value in _walk_json_strings(projected):
        if policy.contains(value):
            raise ProjectionError(f"local path survived projection at {relative}#{pointer or '/'}")
    return (_json_bytes(projected) if transformations else source), transformations, used


def _project_text(
    relative: str,
    source: bytes,
    policy: ForbiddenPolicy,
    approvals: list[TextApproval],
) -> tuple[bytes, list[dict[str, Any]]]:
    try:
        original = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProjectionError(f"selected text artifact is not UTF-8: {relative}") from exc
    projected = original
    transformations: list[dict[str, Any]] = []
    for approval in approvals:
        match = approval.match
        if approval.github_owner_sha256:
            matches = {
                candidate.group(0)
                for candidate in GITHUB_OWNER_URL.finditer(original)
                if _sha256(candidate.group(1).casefold().encode("utf-8"))
                == approval.github_owner_sha256
            }
            if len(matches) != 1:
                raise ProjectionError(
                    f"stale or ambiguous GitHub-owner transform for {relative}: "
                    f"found {len(matches)} matching owner URL bases"
                )
            match = next(iter(matches))
        assert match is not None
        actual = original.count(match)
        if actual != approval.expected_occurrences:
            raise ProjectionError(
                f"stale text transform count for {relative}: {_sha256(match.encode('utf-8'))}: "
                f"{actual} != {approval.expected_occurrences}"
            )
        projected = projected.replace(match, approval.replacement)
        transformations.append(
            {
                "kind": "text_exact",
                "purpose": approval.purpose,
                "original_value_sha256": _sha256(match.encode("utf-8")),
                "replacement": approval.replacement,
                "occurrence_count": actual,
            }
        )
    if policy.contains(projected):
        raise ProjectionError(f"unapproved forbidden content in text artifact: {relative}")
    return projected.encode("utf-8"), transformations


def build_projection(repo_root: Path, scope_path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    lexical_root = repo_root.absolute()
    lexical_scope = scope_path.absolute()
    try:
        lexical_relative = lexical_scope.relative_to(lexical_root).as_posix()
    except ValueError as exc:
        raise ProjectionError("scope config must be inside the repository") from exc
    _assert_no_symlink_components(lexical_root, lexical_relative, "scope config")
    repo_root = repo_root.resolve()
    try:
        scope_relative = scope_path.resolve(strict=True).relative_to(repo_root).as_posix()
    except (OSError, ValueError) as exc:
        raise ProjectionError("scope config must be a regular file inside the repository") from exc
    _assert_regular_inside(repo_root, scope_relative)
    scope = load_scope(repo_root, scope_path)
    selected, manifest = resolve_scope(repo_root, scope)
    policy = _policy(scope)
    _assert_scope_metadata_clean(scope, policy)
    for relative in selected:
        if policy.contains(relative):
            raise ProjectionError(f"forbidden content in selected relative filename: {relative}")
    approvals = _json_approvals(scope, set(selected), policy)
    text_approvals = _text_approvals(scope, set(selected), policy)
    used_approvals: set[tuple[str, str]] = set()
    projected_files: dict[str, bytes] = {}
    records: list[dict[str, Any]] = []
    transform_records: list[dict[str, Any]] = []
    for relative in selected:
        source = (repo_root / relative).read_bytes()
        if b"\x00" in source:
            raise ProjectionError(f"NUL byte in selected text artifact: {relative}")
        if relative.endswith(".json"):
            projected, transformations, used = _project_json(
                relative, source, policy, approvals
            )
            used_approvals.update(used)
        else:
            projected, transformations = _project_text(
                relative, source, policy, text_approvals.get(relative, [])
            )
        projected_files[relative] = projected
        source_sha = _sha256(source)
        export_sha = _sha256(projected)
        record = {
            "path": relative,
            "original_bytes": len(source),
            "original_sha256": source_sha,
            "export_bytes": len(projected),
            "export_sha256": export_sha,
            "transformation_count": len(transformations),
            "transformation_occurrence_count": sum(
                item["occurrence_count"] for item in transformations
            ),
        }
        records.append(record)
        for transformation in transformations:
            transform_records.append(
                {
                    "file": relative,
                    "original_file_sha256": source_sha,
                    "export_file_sha256": export_sha,
                    **transformation,
                }
            )
    unused = set(approvals) - used_approvals
    if unused:
        rendered = ", ".join(f"{path}#{pointer}" for path, pointer in sorted(unused))
        raise ProjectionError(f"stale transform approval(s): {rendered}")
    index_lines = [
        f"{record['path']}\0{record['original_sha256']}\0{record['export_sha256']}"
        for record in records
    ]
    manifest_relative = _safe_relative(scope["paper_manifest"])
    receipt = {
        "schema": SCHEMA,
        "deterministic": True,
        "scope_config": scope_relative,
        "scope_config_sha256": _sha256(scope_path.read_bytes()),
        "paper_manifest": manifest_relative,
        "paper_manifest_sha256": _sha256((repo_root / manifest_relative).read_bytes()),
        "projection_index_sha256": _sha256("\n".join(index_lines).encode("utf-8")),
        "summary": {
            "file_count": len(records),
            "transformed_file_count": len({item["file"] for item in transform_records}),
            "transformation_rule_count": len(transform_records),
            "transformation_occurrence_count": sum(
                item["occurrence_count"] for item in transform_records
            ),
            "unshipped_artifact_count": len(scope["unshipped_artifacts"]),
            "unshipped_artifact_bytes": sum(
                item["original_bytes"] for item in scope["unshipped_artifacts"]
            ),
            "unshipped_artifact_uri_count": sum(
                item["content_addressed_download_uri"] is not None
                for item in scope["unshipped_artifacts"]
            ),
            "manifest_input_exclusion_count": len(
                scope["manifest_input_exclusions"]
            ),
            "unapproved_leak_count": 0,
        },
        "exclusion_policy": scope["excluded_globs"],
        "unshipped_artifacts": copy.deepcopy(scope["unshipped_artifacts"]),
        "manifest_input_exclusions": copy.deepcopy(
            scope["manifest_input_exclusions"]
        ),
        "files": records,
        "transformations": transform_records,
    }
    if policy.contains(json.dumps(receipt, ensure_ascii=False, sort_keys=True)):
        raise ProjectionError("forbidden content leaked into generated receipt metadata")
    return projected_files, receipt


def compact_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": COMPACT_SCHEMA,
        "scope_config": receipt["scope_config"],
        "scope_config_sha256": receipt["scope_config_sha256"],
        "paper_manifest": receipt["paper_manifest"],
        "paper_manifest_sha256": receipt["paper_manifest_sha256"],
        "projection_index_sha256": receipt["projection_index_sha256"],
        "summary": receipt["summary"],
        "unshipped_artifacts": receipt["unshipped_artifacts"],
        "manifest_input_exclusions": receipt["manifest_input_exclusions"],
        "transformations": receipt["transformations"],
    }


def write_projection(output: Path, projected: dict[str, bytes], receipt: dict[str, Any]) -> None:
    if output.exists():
        raise ProjectionError(f"output already exists; choose an empty path: {output}")
    output.mkdir(parents=True)
    for relative, payload in projected.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    (output / RECEIPT_NAME).write_bytes(_json_bytes(receipt))


def check_projection(output: Path, projected: dict[str, bytes], receipt: dict[str, Any]) -> None:
    expected = {**projected, RECEIPT_NAME: _json_bytes(receipt)}
    if output.is_symlink():
        raise ProjectionError(f"projected release directory is a symlink: {output}")
    if not output.is_dir():
        raise ProjectionError(f"projected release directory is missing: {output}")
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ProjectionError(f"symlink found in projected release: {path}")
    actual_paths = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)
        unexpected = sorted(actual_paths - set(expected))
        raise ProjectionError(f"projection file set mismatch; missing={missing}, unexpected={unexpected}")
    for relative, payload in expected.items():
        actual = (output / relative).read_bytes()
        if actual != payload:
            raise ProjectionError(f"projected file is stale or modified: {relative}")


def check_compact_receipt(path: Path, receipt: dict[str, Any]) -> None:
    expected = _json_bytes(compact_receipt(receipt))
    if path.is_symlink():
        raise ProjectionError(f"compact receipt must not be a symlink: {path}")
    if not path.is_file():
        raise ProjectionError(f"compact receipt is missing: {path}")
    if path.read_bytes() != expected:
        raise ProjectionError(f"compact receipt is stale: {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--scope", type=Path, default=Path(DEFAULT_SCOPE))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit", action="store_true", help="validate and summarize without writing")
    mode.add_argument("--check", action="store_true", help="check an existing projected directory")
    parser.add_argument(
        "--check-compact-receipt",
        type=Path,
        help="also compare a tracked compact receipt with the current projection",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    scope_path = args.scope if args.scope.is_absolute() else repo_root / args.scope
    output = args.output if args.output.is_absolute() else repo_root / args.output
    compact_path = args.check_compact_receipt
    if compact_path is not None and not compact_path.is_absolute():
        compact_path = repo_root / compact_path
    try:
        projected, receipt = build_projection(repo_root, scope_path)
        if compact_path is not None:
            check_compact_receipt(compact_path, receipt)
        if args.check:
            check_projection(output, projected, receipt)
            action = "checked"
        elif args.audit:
            action = "audited"
        else:
            write_projection(output, projected, receipt)
            action = "built"
    except ProjectionError as exc:
        print(f"anonymous release failed: {exc}", file=sys.stderr)
        return 1
    summary = receipt["summary"]
    print(
        f"Anonymous release {action}: {summary['file_count']} files, "
        f"{summary['transformed_file_count']} transformed files, "
        f"{summary['transformation_rule_count']} approved rules / "
        f"{summary['transformation_occurrence_count']} replacements, 0 unapproved leaks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
