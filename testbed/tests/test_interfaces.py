"""Unit tests for interface resolution, keys, and preflight (no AI, no network)."""
import tempfile
import unittest
from pathlib import Path

from banter import interfaces, preflight


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class InterfaceTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ifaces = self.root / "platforms"
        # Redirect the single unified tree at the temp dir.
        self._orig = interfaces.PLATFORMS_DIR
        interfaces.PLATFORMS_DIR = self.ifaces

    def tearDown(self) -> None:
        interfaces.PLATFORMS_DIR = self._orig
        self._tmp.cleanup()

    def write_manifest(self, platform: str, interface: str, text: str) -> Path:
        p = self.ifaces / platform / interface / "config.yaml"
        _write(p, text)
        return p


class KeysTests(InterfaceTestBase):
    def test_keys_for_mapping(self):
        self.write_manifest("svc", "sdk", "keys:\n  API_KEY: \"abc\"\n  HOST: \"\"\nprompt: hi\n")
        self.assertEqual(interfaces.keys_for("svc", "sdk"), {"API_KEY": "abc", "HOST": ""})

    def test_keys_for_none(self):
        self.assertEqual(interfaces.keys_for("none", "none"), {})

    def test_resolved_keys_fall_back_to_env(self):
        self.write_manifest("svc", "sdk", "keys:\n  API_KEY: \"\"\nprompt: hi\n")
        got = interfaces._resolved_keys("svc", "sdk", env={"API_KEY": "from-env"})
        self.assertEqual(got, {"API_KEY": "from-env"})


class AccountingFieldsTests(InterfaceTestBase):
    """cli_command / sdk_module drive cli_calls / sdk_calls accounting."""

    def test_resolved_config_surfaces_overrides(self):
        self.write_manifest(
            "svc", "cli",
            "binary: svc_cli-0.1.0-py3-none-any.whl\ncli_command: svc\nprompt: hi\n",
        )
        cfg = interfaces._resolved_config("svc", "cli", 0, None)
        self.assertEqual(cfg["cli_command"], "svc")
        self.assertEqual(cfg["binary"], "svc_cli-0.1.0-py3-none-any.whl")

    def test_resolved_config_defaults_none_when_absent(self):
        self.write_manifest("svc", "cli", "binary: hops\nprompt: hi\n")
        cfg = interfaces._resolved_config("svc", "cli", 0, None)
        self.assertIsNone(cfg["cli_command"])   # falls back to binary in setup()
        self.assertIsNone(cfg["sdk_module"])
        self.assertEqual(cfg["teardown"], [])   # default: nothing to tear down

    def test_resolved_config_surfaces_teardown(self):
        self.write_manifest("svc", "cli", "teardown:\n  - 'echo bye'\nprompt: hi\n")
        cfg = interfaces._resolved_config("svc", "cli", 0, None)
        self.assertEqual(cfg["teardown"], ["echo bye"])


class VersionResolutionTests(InterfaceTestBase):
    def test_base_version_is_zero(self):
        self.write_manifest("svc", "sdk", "prompt: base prompt\n")
        version, _hash = interfaces.variant_for("svc", "sdk")
        self.assertEqual(version, 0)

    def test_version_gt_zero_requires_version_root(self):
        self.write_manifest("svc", "sdk", "prompt: base\n")
        with self.assertRaises(ValueError):
            interfaces.variant_for("svc", "sdk", version=1)

    def test_prompt_frozen_against_version_override(self):
        """The prompt is FROZEN: a version.yaml `prompt:` is ignored — the
        engineer always sees the committed base prompt."""
        self.write_manifest("svc", "sdk", "prompt: base prompt\n")
        vroot = self.root / "session"
        vp = interfaces.version_dir(vroot, "svc", "sdk", 1) / "version.yaml"
        _write(vp, "prompt: improved prompt\n")
        version, _ = interfaces.variant_for("svc", "sdk", version=1, version_root=vroot)
        self.assertEqual(version, 1)
        frag = interfaces._prompt_for("svc", "sdk", 1, vroot)
        self.assertEqual(frag, "base prompt")  # override ignored

    def test_prompt_frozen_against_interface_home_copy(self):
        """The autoresearch flow points the interface home at a per-version copy
        (set_interface_home) whose config.yaml prompt the researcher may edit.
        That edit must be ignored — the prompt comes from the committed base."""
        self.write_manifest("svc", "sdk", "prompt: base prompt\n")
        copy = self.root / "run" / "v1" / "interface"
        _write(copy / "config.yaml", "prompt: hacked prompt\n")
        try:
            interfaces.set_interface_home("svc", "sdk", copy)
            frag = interfaces._prompt_for("svc", "sdk", 0, None)
            self.assertEqual(frag, "base prompt")
        finally:
            interfaces._INTERFACE_HOME.pop(("svc", "sdk"), None)

    def test_prompt_frozen_when_committed_is_empty(self):
        """An intentionally EMPTY committed prompt still freezes to empty — the
        home-redirected copy's prompt must not leak in via the falsy fallthrough."""
        self.write_manifest("svc", "sdk", 'prompt: ""\n')
        copy = self.root / "run" / "v1" / "interface"
        _write(copy / "config.yaml", "prompt: hacked\n")
        try:
            interfaces.set_interface_home("svc", "sdk", copy)
            self.assertEqual(interfaces._prompt_for("svc", "sdk", 0, None), "")
        finally:
            interfaces._INTERFACE_HOME.pop(("svc", "sdk"), None)

    def test_none_interface(self):
        self.assertEqual(interfaces.variant_for("none", "none"), (0, ""))


class SourceFingerprintTests(InterfaceTestBase):
    def _make_iface(self, base: Path, src_body: str) -> Path:
        _write(base / "config.yaml", "prompt: hi\n")
        _write(base / "src" / "python" / "pkg" / "mod.py", src_body)
        return base

    def test_fingerprint_ignores_config_and_artifacts(self):
        a = self._make_iface(self.root / "a", "x = 1\n")
        b = self._make_iface(self.root / "b", "x = 1\n")
        # b differs only in config.yaml (the prompt) + a built wheel + pycache.
        _write(b / "config.yaml", "prompt: TOTALLY DIFFERENT\n")
        _write(b / "thing-0-py3-none-any.whl", "binarygarbage")
        _write(b / "src" / "python" / "pkg" / "__pycache__" / "mod.cpython-312.pyc", "z")
        self.assertEqual(
            interfaces.source_fingerprint(a), interfaces.source_fingerprint(b)
        )

    def test_fingerprint_detects_source_change(self):
        a = self._make_iface(self.root / "a", "x = 1\n")
        b = self._make_iface(self.root / "b", "x = 2\n")  # real source edit
        self.assertNotEqual(
            interfaces.source_fingerprint(a), interfaces.source_fingerprint(b)
        )

    def test_fingerprint_detects_non_prompt_config_change(self):
        """A real config.yaml edit (NOT the prompt) is a source change — the
        researcher is allowed to change install/plumbing, so it must count."""
        a = self._make_iface(self.root / "a", "x = 1\n")
        b = self._make_iface(self.root / "b", "x = 1\n")
        _write(a / "config.yaml", "prompt: hi\nbinary: tool-0.whl\n")
        _write(b / "config.yaml", "prompt: hi\nbinary: tool-9.whl\n")  # plumbing differs
        self.assertNotEqual(
            interfaces.source_fingerprint(a), interfaces.source_fingerprint(b)
        )

    def test_fingerprint_ignores_prompt_only_config_change(self):
        """Only the prompt differs in config.yaml → identical fingerprint."""
        a = self._make_iface(self.root / "a", "x = 1\n")
        b = self._make_iface(self.root / "b", "x = 1\n")
        _write(a / "config.yaml", "prompt: one\nbinary: tool.whl\n")
        _write(b / "config.yaml", "prompt: two\nbinary: tool.whl\n")
        self.assertEqual(
            interfaces.source_fingerprint(a), interfaces.source_fingerprint(b)
        )

    def test_assert_source_changed_blocks_prompt_only_version(self):
        run = self.root / "run"
        v0 = run / "v0" / "interface"
        v1 = run / "v1" / "interface"
        self._make_iface(v0, "x = 1\n")
        self._make_iface(v1, "x = 1\n")            # identical source
        _write(v1 / "config.yaml", "prompt: reworded\n")  # only the prompt moved
        with self.assertRaises(ValueError):
            interfaces.assert_source_changed(v1, "v1")

    def test_assert_source_changed_allows_real_edit(self):
        run = self.root / "run"
        v0 = run / "v0" / "interface"
        v1 = run / "v1" / "interface"
        self._make_iface(v0, "x = 1\n")
        self._make_iface(v1, "x = 2\n")            # real source change
        interfaces.assert_source_changed(v1, "v1")  # must not raise

    def test_assert_source_changed_skips_baseline_and_unlabeled(self):
        run = self.root / "run"
        v0 = run / "v0" / "interface"
        self._make_iface(v0, "x = 1\n")
        interfaces.assert_source_changed(v0, "v0")   # baseline — no constraint
        interfaces.assert_source_changed(v0, None)   # non-autoresearch — no constraint

    def test_assert_source_changed_skips_when_no_prev(self):
        # v1 with no sibling v0 on disk (e.g. a prev_run continuation) — allowed.
        run = self.root / "run"
        v1 = run / "v1" / "interface"
        self._make_iface(v1, "x = 1\n")
        interfaces.assert_source_changed(v1, "v1")   # must not raise


class PreflightTests(InterfaceTestBase):
    def test_none_is_ok(self):
        st = interfaces.preflight("none", "none")
        self.assertTrue(st.ok)

    def test_unknown_type(self):
        st = interfaces.preflight("svc", "bogus")
        self.assertFalse(st.ok)

    def test_missing_config_not_installed(self):
        st = interfaces.preflight("ghost", "cli", auto_build=False)
        self.assertFalse(st.ok)
        self.assertFalse(st.installed)

    def test_sdk_missing_keys_fails_login(self):
        # No auth_command + declared keys → login satisfied only when keys set.
        self.write_manifest("svc", "sdk", "keys:\n  API_KEY: \"\"\nprompt: hi\n")
        st = interfaces.preflight("svc", "sdk", env={})
        self.assertFalse(st.ok)
        self.assertIn("API_KEY", st.missing_keys)
        self.assertIn("setup", st.fix_command)

    def test_sdk_keys_present_passes(self):
        self.write_manifest("svc", "sdk", "keys:\n  API_KEY: \"\"\nprompt: hi\n")
        st = interfaces.preflight("svc", "sdk", env={"API_KEY": "x"})
        self.assertTrue(st.ok)
        self.assertTrue(st.authenticated)

    def test_auth_command_success(self):
        self.write_manifest("svc", "cli", "auth_command: \"true\"\nprompt: hi\n")
        st = interfaces.preflight("svc", "cli", env={})
        self.assertTrue(st.ok)

    def test_auth_command_failure(self):
        self.write_manifest("svc", "cli", "auth_command: \"false\"\nprompt: hi\n")
        st = interfaces.preflight("svc", "cli", env={})
        self.assertFalse(st.ok)
        self.assertFalse(st.authenticated)

    def test_test_command_failure(self):
        self.write_manifest("svc", "cli", "test_command: \"false\"\nprompt: hi\n")
        st = interfaces.preflight("svc", "cli", env={})
        self.assertFalse(st.ok)

    def test_missing_binary_without_build(self):
        self.write_manifest(
            "svc", "cli",
            "binary: tool\nruntime_install:\n  - cp $INTERFACE_DIR/tool .\nprompt: hi\n",
        )
        st = interfaces.preflight("svc", "cli", auto_build=False, env={})
        self.assertFalse(st.ok)
        self.assertFalse(st.installed)


class PreflightModuleTests(InterfaceTestBase):
    def test_none_requirement_passes(self):
        preflight.preflight(
            [preflight.Requirement(platform="none", interface="none")],
            auth="api-key", model="claude-sonnet-4-6", probe_skills=False,
        )

    def test_missing_interface_raises(self):
        with self.assertRaises(preflight.PreflightError):
            preflight.preflight(
                [preflight.Requirement(platform="ghost", interface="cli")],
                auth="api-key", model="claude-sonnet-4-6", probe_skills=False,
            )


if __name__ == "__main__":
    unittest.main()
