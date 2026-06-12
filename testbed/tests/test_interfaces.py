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
        # Redirect BOTH trees at the temp dir: flat config manifests + build homes.
        self.configs = self.root / "configs" / "platforms"
        self.builds = self.root / "build"
        self._orig = (interfaces.CONFIGS_DIR, interfaces.BUILD_DIR)
        interfaces.CONFIGS_DIR = self.configs
        interfaces.BUILD_DIR = self.builds

    def tearDown(self) -> None:
        interfaces.CONFIGS_DIR, interfaces.BUILD_DIR = self._orig
        self._tmp.cleanup()

    def write_manifest(self, platform: str, interface: str, text: str) -> Path:
        p = self.configs / platform / f"{interface}.yaml"
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
        cfg = interfaces._resolved_config("svc", "cli")
        self.assertEqual(cfg["cli_command"], "svc")
        self.assertEqual(cfg["binary"], "svc_cli-0.1.0-py3-none-any.whl")

    def test_resolved_config_defaults_none_when_absent(self):
        self.write_manifest("svc", "cli", "binary: hops\nprompt: hi\n")
        cfg = interfaces._resolved_config("svc", "cli")
        self.assertIsNone(cfg["cli_command"])   # falls back to binary in setup()
        self.assertIsNone(cfg["sdk_module"])
        self.assertEqual(cfg["teardown"], [])   # default: nothing to tear down

    def test_resolved_config_surfaces_teardown(self):
        self.write_manifest("svc", "cli", "teardown:\n  - 'echo bye'\nprompt: hi\n")
        cfg = interfaces._resolved_config("svc", "cli")
        self.assertEqual(cfg["teardown"], ["echo bye"])

    def test_norm_subcommands_string_list_and_none(self):
        # Manifest `cli_subcommand` accepts one service or a list; both normalize
        # to the comma-joined TESTBED_CLI_SUBCOMMAND wire format.
        self.assertEqual(interfaces._norm_subcommands("sagemaker"), "sagemaker")
        self.assertEqual(
            interfaces._norm_subcommands(["sagemaker", "sagemaker-runtime", "s3"]),
            "sagemaker,sagemaker-runtime,s3",
        )
        self.assertIsNone(interfaces._norm_subcommands(None))
        self.assertIsNone(interfaces._norm_subcommands([]))
        self.assertIsNone(interfaces._norm_subcommands("  "))


class BaseCleanTests(InterfaceTestBase):
    """The base .venv must stay free of the interface package so each per-run /
    check venv installs the wheel fresh (console scripts + extras)."""

    def test_dist_name_from_wheel_binary(self):
        # `<dist>-<version>-...whl` → first '-'-delimited token is the dist name.
        self.assertEqual(
            interfaces._interface_dist_name(
                {}, "hopsworks", "hopsworks-0-py3-none-any.whl"),
            "hopsworks",
        )

    def test_dist_name_falls_back_to_sdk_module_then_platform(self):
        self.assertEqual(
            interfaces._interface_dist_name({"sdk_module": "hsfs"}, "hopsworks", None),
            "hsfs",
        )
        self.assertEqual(
            interfaces._interface_dist_name({}, "hopsworks", None), "hopsworks")

    def test_ensure_base_clean_noop_for_none(self):
        # No-op for the null interface — and must not shell out to pip.
        import unittest.mock as mock
        with mock.patch.object(interfaces.subprocess, "run") as run:
            interfaces.ensure_base_clean("none", "none")
        run.assert_not_called()

    def test_ensure_base_clean_uninstalls_when_present(self):
        import unittest.mock as mock
        self.write_manifest(
            "svc", "sdk", "binary: svc_pkg-0-py3-none-any.whl\nprompt: hi\n")
        calls = []

        def fake_run(cmd, *a, **k):
            calls.append(cmd)
            # Emulate `pip show <dist>` → present (exit 0); uninstall → exit 0.
            return mock.Mock(returncode=0)

        with mock.patch.object(interfaces.subprocess, "run", side_effect=fake_run):
            interfaces.ensure_base_clean("svc", "sdk")

        # A `pip show svc_pkg` probe followed by a `pip uninstall -y svc_pkg`.
        self.assertTrue(any("show" in c and "svc_pkg" in c for c in calls))
        self.assertTrue(
            any("uninstall" in c and "-y" in c and "svc_pkg" in c for c in calls))

    def test_ensure_base_clean_skips_uninstall_when_absent(self):
        import unittest.mock as mock
        self.write_manifest(
            "svc", "sdk", "binary: svc_pkg-0-py3-none-any.whl\nprompt: hi\n")
        calls = []

        def fake_run(cmd, *a, **k):
            calls.append(cmd)
            return mock.Mock(returncode=1)  # `pip show` → not present

        with mock.patch.object(interfaces.subprocess, "run", side_effect=fake_run):
            interfaces.ensure_base_clean("svc", "sdk")

        self.assertFalse(any("uninstall" in c for c in calls))


class ResolutionTests(InterfaceTestBase):
    def test_variant_for_returns_hash(self):
        self.write_manifest("svc", "sdk", "prompt: base prompt\n")
        h = interfaces.variant_for("svc", "sdk")
        self.assertTrue(h)
        self.assertEqual(len(h), 8)

    def test_variant_for_unknown_interface_raises(self):
        with self.assertRaises(ValueError):
            interfaces.variant_for("svc", "bogus")

    def test_variant_for_missing_config_raises(self):
        with self.assertRaises(ValueError):
            interfaces.variant_for("ghost", "sdk")

    def test_prompt_comes_from_manifest(self):
        self.write_manifest("svc", "sdk", "prompt: base prompt\n")
        self.assertEqual(interfaces._prompt_for("svc", "sdk"), "base prompt")

    def test_empty_prompt_stays_empty(self):
        # An intentionally EMPTY `prompt:` must not fall through to the
        # auto-generated default.
        self.write_manifest("svc", "sdk", 'prompt: ""\n')
        self.assertEqual(interfaces._prompt_for("svc", "sdk"), "")

    def test_missing_prompt_auto_generates(self):
        self.write_manifest("svc", "sdk", "binary: tool\n")
        self.assertIn("SDK", interfaces._prompt_for("svc", "sdk"))

    def test_sandbox_keys_resolved_from_manifest(self):
        self.write_manifest(
            "svc", "sdk",
            "prompt: base\n"
            "allowed_domains: [api.svc.com]\n"
            "instance_allowlist: [ml.m5.large]\n",
        )
        cfg = interfaces._resolved_config("svc", "sdk")
        self.assertEqual(cfg["allowed_domains"], ["api.svc.com"])
        self.assertEqual(cfg["instance_allowlist"], ["ml.m5.large"])
        self.assertEqual(cfg["sandbox_excluded_commands"], [])

    def test_none_interface(self):
        self.assertEqual(interfaces.variant_for("none", "none"), "")


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
