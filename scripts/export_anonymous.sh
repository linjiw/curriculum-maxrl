#!/bin/bash
# Build a history-free, allowlisted, mechanically anonymized reviewer artifact.
#
# Usage:
#   PYTHON=/path/to/python \
#   PAPER_FIGURE_PYTHON=/path/to/python \
#   TECTONIC_BIN=/path/to/tectonic \
#     bash scripts/export_anonymous.sh /absolute/output/directory
#
# The source tree must be clean. The export reads committed HEAD bytes through
# git archive, never edits canonical artifacts, and refuses to overwrite an
# existing snapshot, tarball, checksum, or report.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

if [ "$#" -ne 1 ]; then
  echo "usage: $0 /absolute/output/directory" >&2
  exit 2
fi

OUTPUT_DIR=$1
case "$OUTPUT_DIR" in
  /*) ;;
  *) echo "output directory must be absolute" >&2; exit 2 ;;
esac

if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
  echo "source tree must be clean before export" >&2
  exit 1
fi

ALLOWLIST="$ROOT/scripts/anonymous_allowlist.txt"
[ -f "$ALLOWLIST" ] && [ ! -L "$ALLOWLIST" ] || {
  echo "missing regular allowlist: $ALLOWLIST" >&2
  exit 1
}

SNAPSHOT_NAME=curriculum-maxrl-anonymous
SNAPSHOT_OUT="$OUTPUT_DIR/$SNAPSHOT_NAME"
ARCHIVE_OUT="$OUTPUT_DIR/$SNAPSHOT_NAME.tar.gz"
CHECKSUM_OUT="$ARCHIVE_OUT.sha256"
REPORT_OUT="$OUTPUT_DIR/ANONYMOUS_EXPORT_REPORT.json"
for target in "$SNAPSHOT_OUT" "$ARCHIVE_OUT" "$CHECKSUM_OUT" "$REPORT_OUT"; do
  [ ! -e "$target" ] || {
    echo "refusing to overwrite existing export target: $target" >&2
    exit 1
  }
done

EXPORT_PYTHON=${EXPORT_PYTHON:-python3}
PYTHON=${PYTHON:-python3}
PAPER_FIGURE_PYTHON=${PAPER_FIGURE_PYTHON:-$PYTHON}
[ -n "${TECTONIC_BIN:-}" ] || TECTONIC_BIN=$(command -v tectonic || true)
for executable in "$EXPORT_PYTHON" "$PYTHON" "$PAPER_FIGURE_PYTHON" "$TECTONIC_BIN"; do
  [ -x "$executable" ] || {
    echo "required executable is unavailable: $executable" >&2
    exit 1
  }
done
command -v git >/dev/null
command -v tar >/dev/null
command -v gzip >/dev/null
command -v sha256sum >/dev/null
command -v pdfinfo >/dev/null
command -v pdftotext >/dev/null

STAGE_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/curriculum-maxrl-anon-export.XXXXXX")
SNAPSHOT="$STAGE_ROOT/$SNAPSHOT_NAME"
mkdir "$SNAPSHOT"
cleanup() {
  status=$?
  rm -rf "$STAGE_ROOT"
  exit "$status"
}
trap cleanup EXIT

mapfile -t ALLOWED_PATHS < <(
  awk 'NF && $1 !~ /^#/ {print $1}' "$ALLOWLIST"
)
[ "${#ALLOWED_PATHS[@]}" -gt 0 ] || {
  echo "anonymous allowlist is empty" >&2
  exit 1
}
for path in "${ALLOWED_PATHS[@]}"; do
  git cat-file -e "HEAD:$path" 2>/dev/null || {
    echo "allowlisted path is absent from HEAD: $path" >&2
    exit 1
  }
done

git archive --format=tar HEAD -- "${ALLOWED_PATHS[@]}" | tar -xf - -C "$SNAPSHOT"
[ ! -e "$SNAPSHOT/.git" ] || {
  echo "Git metadata entered the anonymous snapshot" >&2
  exit 1
}

# This runtime helper is not used by reproduce.sh and contains a host-specific
# default. It is excluded rather than rewritten as executable release code.
UNUSED_HOST_HELPER="$SNAPSHOT/curriculum_maxrl/countdown/audit_e2c_readiness.py"
if [ -e "$UNUSED_HOST_HELPER" ]; then
  [ -f "$UNUSED_HOST_HELPER" ] && [ ! -L "$UNUSED_HOST_HELPER" ] || {
    echo "unexpected helper type: $UNUSED_HOST_HELPER" >&2
    exit 1
  }
  rm "$UNUSED_HOST_HELPER"
fi

# This internal audit intentionally names source commits and therefore must
# not enter the reviewer-facing export it audits.
INTERNAL_ANONYMITY_AUDIT="$SNAPSHOT/paper/ANONYMITY_AUDIT_2026-08-26.md"
if [ -e "$INTERNAL_ANONYMITY_AUDIT" ]; then
  [ -f "$INTERNAL_ANONYMITY_AUDIT" ] && [ ! -L "$INTERNAL_ANONYMITY_AUDIT" ] || {
    echo "unexpected audit type: $INTERNAL_ANONYMITY_AUDIT" >&2
    exit 1
  }
  rm "$INTERNAL_ANONYMITY_AUDIT"
fi

SNAPSHOT="$SNAPSHOT" "$EXPORT_PYTHON" - <<'PY'
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import os
import re

root = Path(os.environ["SNAPSHOT"])

host_path = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:Users|home|data|scratch)/[^/\s\"'`]+"
    r"(?:/[^\s\"'`\)\],;]*)*"
)
author_repo = re.compile(
    r"https?://(?:www\.)?github\.com/[^/\s]+/curriculum-maxrl(?:[^\s\"'`]*)?",
    re.IGNORECASE,
)
author_pages = re.compile(
    r"https?://[^/\s]+\.github\.io/curriculum-maxrl(?:[^\s\"'`]*)?",
    re.IGNORECASE,
)


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def scrub_path(match: re.Match[str]) -> str:
    value = match.group(0)
    parts = value.split("/")
    # Drop the root class and account/mount name, retaining the useful suffix.
    suffix = "/".join(parts[3:])
    return "artifact://host/" + suffix


def scrub_text(text: str) -> tuple[str, int]:
    text, n_path = host_path.subn(scrub_path, text)
    text, n_repo = author_repo.subn("artifact://anonymous-repository", text)
    text, n_pages = author_pages.subn("artifact://anonymous-site", text)
    return text, n_path + n_repo + n_pages


records: list[dict[str, object]] = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.is_symlink():
        continue
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        continue
    scrubbed, count = scrub_text(text)
    if count:
        exported = scrubbed.encode("utf-8")
        path.write_bytes(exported)
        records.append({
            "path": path.relative_to(root).as_posix(),
            "canonical_sha256": digest(data),
            "anonymous_sha256": digest(exported),
            "replacements": count,
            "transformation": "host paths/author-owned repository URLs only",
        })

acrobot = "frontier_rl/examples/acrobot_curriculum_tournament_analysis.json"
if not any(record["path"] == acrobot for record in records):
    raise SystemExit("required Acrobot path scrub did not occur")

# The copied manifest must bind the copied, possibly scrubbed inputs.
manifest_path = root / "paper/results/manifest.json"
manifest_before = manifest_path.read_bytes()
manifest = json.loads(manifest_before)
for section_name in ("figures", "audits"):
    for entry in manifest.get(section_name, {}).values():
        for relative in list(entry.get("checksums", {})):
            target = root / relative
            if not target.is_file() or target.is_symlink():
                raise SystemExit(f"manifest target missing after export: {relative}")
            entry["checksums"][relative] = digest(target.read_bytes())[:16]
manifest_after = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
manifest_path.write_bytes(manifest_after)
if manifest_after != manifest_before:
    records.append({
        "path": "paper/results/manifest.json",
        "canonical_sha256": digest(manifest_before),
        "anonymous_sha256": digest(manifest_after),
        "replacements": 0,
        "transformation": "rebound checksums to anonymous-copy inputs",
    })

# The deposit sidecar in turn binds the copied compact manifest.
deposit_path = root / "paper/PROVENANCE_DEPOSIT.json"
deposit_before = deposit_path.read_bytes()
deposit = json.loads(deposit_before)
for entry in deposit.get("payload", []):
    if entry.get("path") == "paper/results/manifest.json":
        entry["bytes"] = len(manifest_after)
        entry["sha256"] = digest(manifest_after)
deposit_after = (json.dumps(deposit, indent=2, ensure_ascii=False) + "\n").encode()
deposit_path.write_bytes(deposit_after)
if deposit_after != deposit_before:
    records.append({
        "path": "paper/PROVENANCE_DEPOSIT.json",
        "canonical_sha256": digest(deposit_before),
        "anonymous_sha256": digest(deposit_after),
        "replacements": 0,
        "transformation": "rebound compact-manifest byte count and SHA-256",
    })

transform_path = root / "ANONYMIZATION_TRANSFORMS.json"
transform_path.write_text(json.dumps({
    "schema": "curriculum-maxrl/anonymous-transforms/v1",
    "policy": "Only host paths, author-owned repository URLs, and dependent checksum bindings change; scientific values are untouched.",
    "transforms": records,
}, indent=2) + "\n", encoding="utf-8")
PY

# Rebuild and verify from the actual extracted bytes. This is the portable
# contract: no exact machine-specific byte locks are required.
(
  cd "$SNAPSHOT"
  REPRO_MODE=portable \
  PYTHON="$PYTHON" \
  PAPER_FIGURE_PYTHON="$PAPER_FIGURE_PYTHON" \
  TECTONIC_BIN="$TECTONIC_BIN" \
    bash reproduce.sh --build
)

SNAPSHOT="$SNAPSHOT" "$EXPORT_PYTHON" - <<'PY'
from __future__ import annotations

from pathlib import Path
import json
import os
import re
import subprocess

root = Path(os.environ["SNAPSHOT"])

forbidden = {
    "personal absolute path": re.compile(
        rb"(?<![A-Za-z0-9_.-])/(?:Users|home|data|scratch)/[^/\s]+/"
    ),
    "author-owned GitHub repository": re.compile(
        rb"https?://(?:www\.)?github\.com/[^/\s]+/curriculum-maxrl",
        re.IGNORECASE,
    ),
    "author-owned Pages site": re.compile(
        rb"https?://[^/\s]+\.github\.io/curriculum-maxrl",
        re.IGNORECASE,
    ),
}

violations: list[str] = []
file_count = 0
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        violations.append(f"symlink: {path.relative_to(root)}")
        continue
    if not path.is_file():
        continue
    file_count += 1
    data = path.read_bytes()
    for label, pattern in forbidden.items():
        if pattern.search(data):
            violations.append(f"{label}: {path.relative_to(root)}")

pdfs = sorted(root.rglob("*.pdf"))
for pdf in pdfs:
    info = subprocess.run(
        ["pdfinfo", str(pdf)], check=True, text=True, capture_output=True
    ).stdout
    fields = {}
    for line in info.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    for field in ("Author", "Title", "Subject", "Keywords"):
        if fields.get(field):
            violations.append(
                f"PDF {field} metadata: {pdf.relative_to(root)} = {fields[field]!r}"
            )
    text = subprocess.run(
        ["pdftotext", str(pdf), "-"], check=True, capture_output=True
    ).stdout
    for label, pattern in forbidden.items():
        if pattern.search(text):
            violations.append(f"{label} in PDF text: {pdf.relative_to(root)}")

if violations:
    raise SystemExit("anonymous export scan failed:\n" + "\n".join(violations))

report = {
    "schema": "curriculum-maxrl/anonymous-export-report/v1",
    "history_free": not (root / ".git").exists(),
    "allowlist": "committed scripts/anonymous_allowlist.txt",
    "portable_reproduce_build": "passed",
    "file_count": file_count,
    "pdf_count": len(pdfs),
    "scans": {
        "personal_absolute_paths": "passed",
        "author_owned_repository_urls": "passed",
        "pdf_identity_metadata": "passed",
        "pdf_text": "passed",
        "symlinks": "passed",
    },
    "doi": None,
    "upload_status": "not_uploaded_pi_owned",
}
(root / "ANONYMOUS_EXPORT_REPORT.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
PY

mkdir -p "$OUTPUT_DIR"
(
  cd "$STAGE_ROOT"
  tar --sort=name --mtime=@1786718220 --owner=0 --group=0 --numeric-owner \
      -cf - "$SNAPSHOT_NAME" | gzip -n > "$ARCHIVE_OUT"
)
(
  cd "$OUTPUT_DIR"
  sha256sum "$SNAPSHOT_NAME.tar.gz" > "$SNAPSHOT_NAME.tar.gz.sha256"
)
cp "$SNAPSHOT/ANONYMOUS_EXPORT_REPORT.json" "$REPORT_OUT"
mv "$SNAPSHOT" "$SNAPSHOT_OUT"

echo "anonymous export passed"
echo "snapshot: $SNAPSHOT_OUT"
echo "archive:  $ARCHIVE_OUT"
cat "$CHECKSUM_OUT"
