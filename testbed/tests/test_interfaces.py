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

    def test_session_local_version_overrides_prompt(self):
        self.write_manifest("svc", "sdk", "prompt: base prompt\n")
        vroot = self.root / "session"
        vp = interfaces.version_dir(vroot, "svc", "sdk", 1) / "version.yaml"
        _write(vp, "prompt: improved prompt\n")
        version, _ = interfaces.variant_for("svc", "sdk", version=1, version_root=vroot)
        self.assertEqual(version, 1)
        frag = interfaces._prompt_for("svc", "sdk", 1, vroot)
        self.assertEqual(frag, "improved prompt")

    def test_none_interface(self):
        self.assertEqual(interfaces.variant_for("none", "none"), (0, ""))


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
