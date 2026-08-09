from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from curriculum_maxrl.analysis import anonymous_source_bundle as bundle_module
from curriculum_maxrl.analysis.anonymous_source_bundle import (
    RECEIPT_NAME,
    SCOPE_SCHEMA,
    BundleError,
    build_bundle,
    main,
    verify_export,
    write_bundle,
    write_deterministic_archive,
)


class AnonymousSourceBundleTests(unittest.TestCase):
    def setUp(self):
        synthetic_digest = hashlib.sha256(self.owner().encode()).hexdigest()
        self.identity_policy_patch = mock.patch.object(
            bundle_module, "EXPECTED_IDENTITY_SHA256", (synthetic_digest,)
        )
        self.identity_policy_patch.start()

    def tearDown(self):
        self.identity_policy_patch.stop()

    @staticmethod
    def private_path() -> str:
        return "/" + "Users" + "/alice/private.json"

    @staticmethod
    def owner() -> str:
        return "synthetic-reviewer"

    def make_repo(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "README.md").write_text("public\n", encoding="utf-8")
        (root / "artifact.json").write_text(
            json.dumps({"artifact": self.private_path()}, indent=2) + "\n",
            encoding="utf-8",
        )
        (root / "reproduce.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        omitted_payloads = {
            "history_aborted.md": b"historical protocol\n",
            "lock_aborted.json": b'{"historical": true}\n',
        }
        for relative, payload in omitted_payloads.items():
            (root / relative).write_bytes(payload)
        scope = {
            "schema": SCOPE_SCHEMA,
            "expected_selected_file_count": 3,
            "archive_root": "anonymous-test-bundle",
            "groups": [
                {
                    "name": "payload",
                    "expected_file_count": 3,
                    "paths": ["README.md", "artifact.json", "reproduce.sh"],
                }
            ],
            "unshipped_artifacts": [],
            "omissions": [
                {
                    "path": relative,
                    "original_bytes": len(payload),
                    "original_sha256": hashlib.sha256(payload).hexdigest(),
                    "reason": "synthetic historical witness",
                }
                for relative, payload in sorted(omitted_payloads.items())
            ],
            "forbidden_selection_globs": [
                "*aborted*",
                "*invalid*",
                "frontier_rl/examples/*procurl_selection*",
            ],
            "content_policy": {
                "forbidden_prefixes": [
                    "/" + "Users" + "/",
                    "/" + "home" + "/",
                    "/" + "tmp" + "/",
                    "/" + "private/tmp" + "/",
                    "/" + "var/folders" + "/",
                    "/" + "root" + "/",
                    "/" + "Volumes" + "/",
                ],
                "forbidden_literals": ["file" + "://"],
                "forbidden_identity_sha256": [
                    hashlib.sha256(self.owner().encode()).hexdigest()
                ],
            },
            "approved_json_transforms": [
                {
                    "file": "artifact.json",
                    "json_pointer": "/artifact",
                    "replacement": "<REPO_ROOT>/artifact.json",
                }
            ],
            "approved_text_transforms": [],
            "executable_paths": ["reproduce.sh"],
        }
        scope_path = root / "scope.json"
        scope_path.write_text(
            json.dumps(scope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return temporary, root, scope_path

    def test_build_is_nonmutating_receipted_and_deterministic(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        source_before = (root / "artifact.json").read_bytes()
        projected_a, receipt_a, executables_a = build_bundle(root, scope_path)
        projected_b, receipt_b, executables_b = build_bundle(root, scope_path)
        self.assertEqual(projected_a, projected_b)
        self.assertEqual(receipt_a, receipt_b)
        self.assertEqual(executables_a, executables_b)
        self.assertEqual((root / "artifact.json").read_bytes(), source_before)
        self.assertNotEqual(
            receipt_a["files"][1]["original_sha256"],
            receipt_a["files"][1]["export_sha256"],
        )
        self.assertEqual(receipt_a["summary"]["omitted_witness_count"], 2)
        self.assertEqual(receipt_a["summary"]["unshipped_artifact_count"], 0)

        output_a, output_b = root / "out-a", root / "out-b"
        archive_a, archive_b = root / "a.tar.gz", root / "b.tar.gz"
        write_bundle(output_a, projected_a, receipt_a, executables_a)
        write_bundle(output_b, projected_b, receipt_b, executables_b)
        verify_export(output_a)
        verify_export(output_b)
        sha_a = write_deterministic_archive(output_a, archive_a, receipt_a)
        sha_b = write_deterministic_archive(output_b, archive_b, receipt_b)
        self.assertEqual(sha_a, sha_b)
        self.assertEqual(archive_a.read_bytes(), archive_b.read_bytes())
        with tarfile.open(archive_a, "r:gz") as archive:
            members = archive.getmembers()
        self.assertEqual(len(members), 4)
        self.assertTrue(all(member.mtime == 0 for member in members))
        self.assertTrue(all(member.uid == member.gid == 0 for member in members))
        modes = {Path(member.name).name: member.mode for member in members}
        self.assertEqual(modes["reproduce.sh"], 0o755)
        self.assertEqual(modes[RECEIPT_NAME], 0o644)

    def test_duplicate_json_key_fails_closed(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        private = self.private_path()
        (root / "artifact.json").write_text(
            '{"artifact": "' + private + '", "artifact": "public"}\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(BundleError, "duplicate JSON key"):
            build_bundle(root, scope_path)

    def test_duplicate_jsonl_key_fails_closed(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        (root / "records.jsonl").write_text('{"x": 1, "x": 2}\n', encoding="utf-8")
        scope = json.loads(scope_path.read_text(encoding="utf-8"))
        scope["groups"][0]["paths"].remove("README.md")
        scope["groups"][0]["paths"].append("records.jsonl")
        scope["groups"][0]["paths"].sort()
        scope_path.write_text(json.dumps(scope), encoding="utf-8")
        with self.assertRaisesRegex(BundleError, "duplicate JSON key"):
            build_bundle(root, scope_path)

    def test_unapproved_identity_fails_closed(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        (root / "README.md").write_text(
            "reviewed by " + self.owner() + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(BundleError, "forbidden content"):
            build_bundle(root, scope_path)

    def test_selected_source_symlink_fails_closed(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        target = root / "real-readme.md"
        target.write_text("public\n", encoding="utf-8")
        (root / "README.md").unlink()
        (root / "README.md").symlink_to(target.name)
        with self.assertRaisesRegex(BundleError, "symlink component"):
            build_bundle(root, scope_path)

    def test_forbidden_active_path_rejected_before_read(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        scope = json.loads(scope_path.read_text(encoding="utf-8"))
        active = "frontier_rl/examples/acrobot_procurl_selection_confirmatory.json"
        scope["groups"][0]["paths"].append(active)
        scope["groups"][0]["paths"].sort()
        scope["groups"][0]["expected_file_count"] = 4
        scope["expected_selected_file_count"] = 4
        scope_path.write_text(json.dumps(scope), encoding="utf-8")
        with self.assertRaisesRegex(BundleError, "forbidden before any read"):
            build_bundle(root, scope_path)

    def test_forbidden_active_path_cannot_be_smuggled_as_omission(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        scope = json.loads(scope_path.read_text(encoding="utf-8"))
        scope["omissions"][0]["path"] = (
            "frontier_rl/examples/acrobot_procurl_selection_confirmatory.json"
        )
        scope_path.write_text(json.dumps(scope), encoding="utf-8")
        with self.assertRaisesRegex(BundleError, "forbidden before any read"):
            build_bundle(root, scope_path)

        temporary2, root2, scope_path2 = self.make_repo()
        self.addCleanup(temporary2.cleanup)
        projected, receipt, executables = build_bundle(root2, scope_path2)
        output = root2 / "out"
        write_bundle(output, projected, receipt, executables)
        receipt["omissions"][0]["path"] = (
            "frontier_rl/examples/acrobot_procurl_selection_confirmatory.json"
        )
        (output / RECEIPT_NAME).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(BundleError, "forbidden before any read"):
            verify_export(output)

    def test_unshipped_artifact_is_declared_without_reading_or_shipping(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        scope = json.loads(scope_path.read_text(encoding="utf-8"))
        relative = (
            "frontier_rl/examples/acrobot_procurl_selection_confirmatory.json"
        )
        scope["unshipped_artifacts"] = [
            {
                "path": relative,
                "original_bytes": 1374886097,
                "original_sha256": "a" * 64,
                "availability": "external-content-addressed-no-public-uri",
                "content_addressed_download_uri": None,
                "reason": "synthetic unshipped raw declaration",
            }
        ]
        scope_path.write_text(json.dumps(scope), encoding="utf-8")
        projected, receipt, executables = build_bundle(root, scope_path)
        self.assertFalse((root / relative).exists())
        self.assertEqual(receipt["summary"]["unshipped_artifact_count"], 1)
        output = root / "out"
        write_bundle(output, projected, receipt, executables)
        verify_export(output)
        self.assertFalse((output / relative).exists())

        receipt["unshipped_artifacts"][0]["content_addressed_download_uri"] = (
            "https://example.invalid/raw"
        )
        (output / RECEIPT_NAME).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(BundleError, "download URI"):
            verify_export(output)

    def test_generated_and_tampered_receipt_identity_fails_closed(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        scope = json.loads(scope_path.read_text(encoding="utf-8"))
        scope["omissions"][0]["reason"] = "reviewed by " + self.owner()
        scope_path.write_text(json.dumps(scope), encoding="utf-8")
        with self.assertRaisesRegex(BundleError, "receipt metadata"):
            build_bundle(root, scope_path)

        temporary_policy, root_policy, scope_path_policy = self.make_repo()
        self.addCleanup(temporary_policy.cleanup)
        scope_policy = json.loads(scope_path_policy.read_text(encoding="utf-8"))
        scope_policy["content_policy"]["forbidden_identity_sha256"] = []
        scope_path_policy.write_text(json.dumps(scope_policy), encoding="utf-8")
        with self.assertRaisesRegex(BundleError, "forbidden-identity inventory"):
            build_bundle(root_policy, scope_path_policy)

        temporary2, root2, scope_path2 = self.make_repo()
        self.addCleanup(temporary2.cleanup)
        projected, receipt, executables = build_bundle(root2, scope_path2)
        output = root2 / "out"
        write_bundle(output, projected, receipt, executables)
        receipt["content_policy"]["forbidden_identity_sha256"] = []
        (output / RECEIPT_NAME).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(BundleError, "forbidden-identity inventory"):
            verify_export(output)
        receipt["content_policy"]["forbidden_identity_sha256"] = [
            hashlib.sha256(self.owner().encode()).hexdigest()
        ]
        receipt["scientific_hash_policy"] = "reviewed by " + self.owner()
        (output / RECEIPT_NAME).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(BundleError, "receipt metadata"):
            verify_export(output)

    def test_stale_transform_count_fails_closed(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        scope = json.loads(scope_path.read_text(encoding="utf-8"))
        scope["approved_text_transforms"] = [
            {
                "kind": "exact",
                "file": "README.md",
                "match": "public",
                "replacement": "released",
                "expected_occurrences": 2,
                "purpose": "synthetic",
            }
        ]
        scope_path.write_text(json.dumps(scope), encoding="utf-8")
        with self.assertRaisesRegex(BundleError, "stale exact transform count"):
            build_bundle(root, scope_path)

    def test_verify_rejects_extra_tamper_and_output_symlink(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        projected, receipt, executables = build_bundle(root, scope_path)
        output = root / "out"
        write_bundle(output, projected, receipt, executables)
        (output / "extra.txt").write_text("extra\n", encoding="utf-8")
        with self.assertRaisesRegex(BundleError, "file set mismatch"):
            verify_export(output)
        (output / "extra.txt").unlink()
        (output / "README.md").write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(BundleError, "differs from receipt"):
            verify_export(output)
        (output / "README.md").write_bytes(projected["README.md"])
        target = root / "same.json"
        target.write_bytes(projected["artifact.json"])
        (output / "artifact.json").unlink()
        (output / "artifact.json").symlink_to(target)
        with self.assertRaisesRegex(BundleError, "symlink"):
            verify_export(output)

    def test_cli_rejects_symlinked_bundle_root(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        projected, receipt, executables = build_bundle(root, scope_path)
        output = root / "out"
        write_bundle(output, projected, receipt, executables)
        alias = root / "bundle-alias"
        alias.symlink_to(output, target_is_directory=True)
        with self.assertRaisesRegex(BundleError, "symlink"):
            verify_export(alias)
        errors = io.StringIO()
        with redirect_stderr(errors):
            status = main(["--verify-export", "--bundle-root", str(alias)])
        self.assertEqual(status, 1)
        self.assertIn("symlink", errors.getvalue())

    def test_scope_and_receipt_duplicate_keys_fail_closed(self):
        temporary, root, scope_path = self.make_repo()
        self.addCleanup(temporary.cleanup)
        text = scope_path.read_text(encoding="utf-8")
        scope_path.write_text(text.replace("{", '{"schema":"duplicate",', 1), encoding="utf-8")
        with self.assertRaisesRegex(BundleError, "duplicate JSON key"):
            build_bundle(root, scope_path)

        temporary2, root2, scope_path2 = self.make_repo()
        self.addCleanup(temporary2.cleanup)
        projected, receipt, executables = build_bundle(root2, scope_path2)
        output = root2 / "out"
        write_bundle(output, projected, receipt, executables)
        receipt_path = output / RECEIPT_NAME
        receipt_text = receipt_path.read_text(encoding="utf-8")
        receipt_path.write_text(
            receipt_text.replace("{", '{"schema":"duplicate",', 1),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(BundleError, "duplicate JSON key"):
            verify_export(output)


if __name__ == "__main__":
    unittest.main()
