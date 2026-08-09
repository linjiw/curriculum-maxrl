from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from curriculum_maxrl.analysis.anonymous_release import (
    ProjectionError,
    SCOPE_SCHEMA,
    build_projection,
    check_compact_receipt,
    check_projection,
    compact_receipt,
    write_projection,
)


class AnonymousReleaseUnitTests(unittest.TestCase):
    @staticmethod
    def owner_identity() -> str:
        return "synthetic-reviewer"

    def make_repo(self, *, text: str = "public\n", json_path: str = "/Users/alice/repo/raw.json"):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "README.md").write_text(text, encoding="utf-8")
        (root / "artifact.json").write_text(
            json.dumps({"artifact": json_path}, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "manuscript": {},
            "results": {"toy": {"inputs": ["artifact.json"], "checksums": {}}},
            "figures": {},
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        scope = {
            "schema": SCOPE_SCHEMA,
            "paper_manifest": "manifest.json",
            "entrypoints": ["README.md"],
            "manifest_expansion": {
                "include_result_inputs": True,
                "include_figure_inputs": True,
            },
            "unshipped_artifacts": [],
            "manifest_input_exclusions": [],
            "forbidden_prefixes": [
                "/Users/",
                "/home/",
                "/tmp/",
                "/private/tmp/",
                "/var/folders/",
                "/root/",
                "/Volumes/",
            ],
            "forbidden_literals": ["file://"],
            "forbidden_identity_sha256": [
                hashlib.sha256(self.owner_identity().encode("utf-8")).hexdigest()
            ],
            "approved_json_transforms": [
                {
                    "file": "artifact.json",
                    "json_pointer": "/artifact",
                    "replacement": "<REPO_ROOT>/raw.json",
                }
            ],
            "approved_text_transforms": [],
            "excluded_globs": [
                {"glob": "*INVALID*", "reason": "invalid branch"},
                {"glob": "*.log", "reason": "build log"},
            ],
        }
        scope_path = root / "scope.json"
        scope_path.write_text(json.dumps(scope, indent=2) + "\n", encoding="utf-8")
        return temporary, root, scope_path

    def test_projection_is_non_mutating_deterministic_and_receipted(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        original = (root / "artifact.json").read_bytes()
        projected_a, receipt_a = build_projection(root, scope_path)
        projected_b, receipt_b = build_projection(root, scope_path)
        self.assertEqual(projected_a, projected_b)
        self.assertEqual(receipt_a, receipt_b)
        self.assertEqual((root / "artifact.json").read_bytes(), original)
        exported = json.loads(projected_a["artifact.json"])
        self.assertEqual(exported["artifact"], "<REPO_ROOT>/raw.json")
        self.assertEqual(receipt_a["summary"]["transformation_rule_count"], 1)
        self.assertEqual(receipt_a["summary"]["transformation_occurrence_count"], 1)
        transform = receipt_a["transformations"][0]
        self.assertEqual(transform["json_pointer"], "/artifact")
        self.assertEqual(transform["replacement"], "<REPO_ROOT>/raw.json")
        self.assertEqual(transform["occurrence_count"], 1)
        self.assertNotEqual(transform["original_file_sha256"], transform["export_file_sha256"])
        self.assertNotIn("alice", json.dumps(compact_receipt(receipt_a)))

    def test_manifest_input_can_be_digest_bound_and_unshipped_without_reading(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        relative = "external/raw.json"
        digest = "a" * 64
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["results"]["external"] = {
            "inputs": [relative],
            "checksums": {relative: digest[:16]},
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        scope = json.loads(scope_path.read_text(encoding="utf-8"))
        scope["unshipped_artifacts"] = [
            {
                "path": relative,
                "original_bytes": 123,
                "original_sha256": digest,
                "availability": "external-content-addressed-no-public-uri",
                "content_addressed_download_uri": None,
                "reason": "synthetic external artifact",
            }
        ]
        scope["manifest_input_exclusions"] = [
            {"path": relative, "reason": "synthetic compact omission"}
        ]
        scope["excluded_globs"].append(
            {"glob": relative, "reason": "synthetic compact omission"}
        )
        scope_path.write_text(json.dumps(scope), encoding="utf-8")
        projected, receipt = build_projection(root, scope_path)
        self.assertNotIn(relative, projected)
        self.assertFalse((root / relative).exists())
        self.assertEqual(receipt["summary"]["unshipped_artifact_count"], 1)
        self.assertEqual(receipt["summary"]["manifest_input_exclusion_count"], 1)
        self.assertEqual(receipt["summary"]["unshipped_artifact_uri_count"], 0)
        self.assertEqual(
            compact_receipt(receipt)["unshipped_artifacts"][0][
                "content_addressed_download_uri"
            ],
            None,
        )

        scope["unshipped_artifacts"][0]["original_sha256"] = "b" * 64
        scope_path.write_text(json.dumps(scope), encoding="utf-8")
        with self.assertRaisesRegex(ProjectionError, "manifest/declaration checksum"):
            build_projection(root, scope_path)

    def test_stale_manifest_input_exclusion_fails_closed(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        scope = json.loads(scope_path.read_text(encoding="utf-8"))
        relative = "external/unused.json"
        scope["unshipped_artifacts"] = [
            {
                "path": relative,
                "original_bytes": 1,
                "original_sha256": "c" * 64,
                "availability": "external-content-addressed-no-public-uri",
                "content_addressed_download_uri": None,
                "reason": "synthetic unused artifact",
            }
        ]
        scope["manifest_input_exclusions"] = [
            {"path": relative, "reason": "synthetic stale omission"}
        ]
        scope["excluded_globs"].append(
            {"glob": relative, "reason": "synthetic stale omission"}
        )
        scope_path.write_text(json.dumps(scope), encoding="utf-8")
        with self.assertRaisesRegex(ProjectionError, "inventory is stale"):
            build_projection(root, scope_path)

    def test_procurl_raw_paths_remain_never_read_after_scope_drift(self):
        raw_paths = (
            "frontier_rl/examples/acrobot_procurl_selection_confirmatory.json",
            "frontier_rl/examples/acrobot_procurl_selection_development.json",
            "frontier_rl/examples/acrobot_procurl_selection_quick_analysis.json",
        )
        for relative in raw_paths:
            with self.subTest(relative=relative):
                temporary, root, scope_path = self.make_repo()
                self.addCleanup(temporary.cleanup)
                payload = b'{"synthetic": true}\n'
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                manifest_path = root / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["results"]["scope_drift"] = {
                    "inputs": [relative],
                    "checksums": {
                        relative: hashlib.sha256(payload).hexdigest()
                    },
                }
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                scope = json.loads(scope_path.read_text(encoding="utf-8"))
                scope["unshipped_artifacts"] = []
                scope["manifest_input_exclusions"] = []
                scope["excluded_globs"] = [
                    item
                    for item in scope["excluded_globs"]
                    if not Path(relative).match(item["glob"])
                ]
                scope_path.write_text(json.dumps(scope), encoding="utf-8")
                with self.assertRaisesRegex(
                    ProjectionError, "forbidden before any read"
                ):
                    build_projection(root, scope_path)

    def test_unapproved_json_leak_fails_closed(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        data = json.loads((root / "artifact.json").read_text())
        data["new_path"] = "/home/bob/private.json"
        (root / "artifact.json").write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ProjectionError, "unapproved local path"):
            build_projection(root, scope_path)

    def test_root_json_path_fails_closed(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        data = json.loads((root / "artifact.json").read_text())
        data["reviewer"] = "/root/private/reviewer"
        (root / "artifact.json").write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ProjectionError, "unapproved local path"):
            build_projection(root, scope_path)

    def test_local_path_in_json_key_fails_closed(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        data = json.loads((root / "artifact.json").read_text())
        data["/Users/alice/private-key"] = "value"
        (root / "artifact.json").write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ProjectionError, "JSON key"):
            build_projection(root, scope_path)

    def test_text_leak_fails_closed(self):
        temporary, root, scope_path = self.make_repo(text="see /private/tmp/secret/run.json\n")
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(ProjectionError, "text artifact"):
            build_projection(root, scope_path)

    def test_tmp_text_path_fails_closed(self):
        temporary, root, scope_path = self.make_repo(text="output: /tmp/private.json\n")
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(ProjectionError, "text artifact"):
            build_projection(root, scope_path)

    def test_owner_url_and_undeclared_identity_fail_closed(self):
        owner = self.owner_identity()
        for text in (f"https://github.com/{owner}/private\n", f"reviewer {owner}\n"):
            with self.subTest(text=text):
                temporary, root, scope_path = self.make_repo(text=text)
                self.addCleanup(temporary.cleanup)
                with self.assertRaisesRegex(ProjectionError, "text artifact"):
                    build_projection(root, scope_path)

    def test_third_party_owner_is_retained(self):
        temporary, root, scope_path = self.make_repo(
            text="https://github.com/tajwarfahim/maxrl\n"
        )
        self.addCleanup(temporary.cleanup)
        projected, _ = build_projection(root, scope_path)
        self.assertIn(b"tajwarfahim", projected["README.md"])

    def test_windows_user_path_fails_closed(self):
        temporary, root, scope_path = self.make_repo(text="C:\\Users\\alice\\secret.json\n")
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(ProjectionError, "text artifact"):
            build_projection(root, scope_path)

    def test_windows_unc_path_fails_closed(self):
        temporary, root, scope_path = self.make_repo(
            text="network: " + r"\\server\Users\alice\secret.json" + "\n"
        )
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(ProjectionError, "text artifact"):
            build_projection(root, scope_path)

    def test_local_file_url_is_case_insensitive(self):
        temporary, root, scope_path = self.make_repo(text="FILE://private/path\n")
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(ProjectionError, "text artifact"):
            build_projection(root, scope_path)

    def test_unconfigured_generic_posix_path_is_outside_claimed_contract(self):
        temporary, root, scope_path = self.make_repo(text="public tool: /opt/public/bin/run\n")
        self.addCleanup(temporary.cleanup)
        projected, _ = build_projection(root, scope_path)
        self.assertIn(b"/opt/public/bin/run", projected["README.md"])

    def test_stale_text_occurrence_count_fails_closed(self):
        temporary, root, scope_path = self.make_repo(text="output: /tmp/private.json\n")
        self.addCleanup(temporary.cleanup)
        scope = json.loads(scope_path.read_text())
        scope["approved_text_transforms"] = [
            {
                "file": "README.md",
                "match": "/tmp/private.json",
                "replacement": "<TMP_DIR>/private.json",
                "expected_occurrences": 2,
            }
        ]
        scope_path.write_text(json.dumps(scope), encoding="utf-8")
        with self.assertRaisesRegex(ProjectionError, "stale text transform count"):
            build_projection(root, scope_path)

    def test_duplicate_json_key_cannot_hide_forbidden_value(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        (root / "artifact.json").write_text(
            '{"artifact":"/Users/alice/private", "artifact":"relative.json"}\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ProjectionError, "duplicate JSON key"):
            build_projection(root, scope_path)

    def test_forbidden_identity_in_scope_metadata_fails_closed(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        scope = json.loads(scope_path.read_text())
        scope["excluded_globs"][0]["reason"] = f"reviewed by {self.owner_identity()}"
        scope_path.write_text(json.dumps(scope), encoding="utf-8")
        with self.assertRaisesRegex(ProjectionError, "scope metadata"):
            build_projection(root, scope_path)

    def test_forbidden_identity_in_manifest_selected_filename_fails_closed(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        identity_file = f"{self.owner_identity()}.json"
        (root / identity_file).write_text("{}\n", encoding="utf-8")
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["results"]["toy"]["inputs"].append(identity_file)
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ProjectionError, "selected relative filename"):
            build_projection(root, scope_path)

    def test_forbidden_identity_in_replacement_fails_closed(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        scope = json.loads(scope_path.read_text())
        scope["approved_json_transforms"][0]["replacement"] = (
            f"owner/{self.owner_identity()}/raw.json"
        )
        scope_path.write_text(json.dumps(scope), encoding="utf-8")
        with self.assertRaisesRegex(ProjectionError, "scope metadata|unsafe transform replacement"):
            build_projection(root, scope_path)

    def test_stale_approval_fails_closed(self):
        temporary, root, scope_path = self.make_repo(json_path="relative/raw.json")
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(ProjectionError, "stale transform approval"):
            build_projection(root, scope_path)

    def test_selected_invalid_or_log_artifact_is_rejected(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        scope = json.loads(scope_path.read_text())
        scope["entrypoints"].append("run_INVALID.json")
        (root / "run_INVALID.json").write_text("{}\n", encoding="utf-8")
        scope_path.write_text(json.dumps(scope), encoding="utf-8")
        with self.assertRaisesRegex(ProjectionError, "covered by an exclusion"):
            build_projection(root, scope_path)

    def test_written_projection_checks_exact_file_set_and_bytes(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        projected, receipt = build_projection(root, scope_path)
        output = root / "export"
        write_projection(output, projected, receipt)
        check_projection(output, projected, receipt)
        (output / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
        with self.assertRaisesRegex(ProjectionError, "file set mismatch"):
            check_projection(output, projected, receipt)

    def test_selected_input_with_symlink_ancestor_is_rejected(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        outside_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(outside_tmp.cleanup)
        outside = Path(outside_tmp.name)
        (outside / "artifact.json").write_text("{}\n", encoding="utf-8")
        (root / "linked").symlink_to(outside, target_is_directory=True)
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["results"]["toy"]["inputs"] = ["linked/artifact.json"]
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ProjectionError, "symlink component"):
            build_projection(root, scope_path)

    def test_output_symlink_is_rejected_even_when_target_bytes_match(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        projected, receipt = build_projection(root, scope_path)
        output = root / "export"
        write_projection(output, projected, receipt)
        target = root / "same-bytes.json"
        target.write_bytes(projected["artifact.json"])
        exported = output / "artifact.json"
        exported.unlink()
        exported.symlink_to(target)
        with self.assertRaisesRegex(ProjectionError, "symlink"):
            check_projection(output, projected, receipt)

    def test_symlinked_scope_config_is_rejected_even_when_target_is_internal(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        alias = root / "scope-alias.json"
        alias.symlink_to(scope_path.name)
        with self.assertRaisesRegex(ProjectionError, "scope config has a symlink"):
            build_projection(root, alias)

    def test_symlinked_compact_receipt_is_rejected_even_when_bytes_match(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        _, receipt = build_projection(root, scope_path)
        target = root / "compact-real.json"
        target.write_text(
            json.dumps(compact_receipt(receipt), indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        alias = root / "compact-alias.json"
        alias.symlink_to(target.name)
        with self.assertRaisesRegex(ProjectionError, "must not be a symlink"):
            check_compact_receipt(alias, receipt)


class AnonymousReleaseRepositoryIntegrationTest(unittest.TestCase):
    def test_current_release_scope_is_complete_and_leak_free_after_projection(self):
        repo_root = Path(__file__).resolve().parents[2]
        projected, receipt = build_projection(
            repo_root, repo_root / "anonymous_release_scope.json"
        )
        self.assertGreaterEqual(receipt["summary"]["file_count"], 40)
        self.assertEqual(receipt["summary"]["transformed_file_count"], 10)
        self.assertEqual(receipt["summary"]["transformation_rule_count"], 16)
        self.assertEqual(receipt["summary"]["transformation_occurrence_count"], 31)
        self.assertEqual(receipt["summary"]["unshipped_artifact_count"], 2)
        self.assertEqual(receipt["summary"]["manifest_input_exclusion_count"], 1)
        self.assertEqual(receipt["summary"]["unshipped_artifact_uri_count"], 0)
        self.assertEqual(receipt["summary"]["unapproved_leak_count"], 0)
        self.assertNotIn(
            "frontier_rl/examples/acrobot_procurl_selection_confirmatory.json", projected
        )
        self.assertIn(
            b"github.com/tajwarfahim/maxrl", projected["MAXRL_SOURCE_AUDIT.md"]
        )
        self.assertNotIn(b"bash reproduce.sh", projected["README.md"])
        self.assertIn(b"evidence-only double-blind supplement", projected["README.md"])


if __name__ == "__main__":
    unittest.main()
