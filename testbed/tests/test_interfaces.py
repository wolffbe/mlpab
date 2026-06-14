"""Unit tests for interface resolution, keys, and preflight (no AI, no network)."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mlpab import interfaces, preflight


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
        self.write_manifest("svc", "sdk", 'keys:\n  API_KEY: "abc"\n  HOST: ""\nprompt: hi\n')
        self.assertEqual(interfaces.keys_for("svc", "sdk"), {"API_KEY": "abc", "HOST": ""})

    def test_keys_for_none(self):
        self.assertEqual(interfaces.keys_for("none", "none"), {})

    def test_resolved_keys_fall_back_to_env(self):
        self.write_manifest("svc", "sdk", 'keys:\n  API_KEY: ""\nprompt: hi\n')
        got = interfaces._resolved_keys("svc", "sdk", env={"API_KEY": "from-env"})
        self.assertEqual(got, {"API_KEY": "from-env"})

    def test_optional_key_excluded_from_missing(self):
        # An `optional: true` key is declared (so it's resolved/injected when a
        # value exists) but its absence must NOT trip the availability gate — a
        # platform setup step provisions it at run time (e.g. SAGEMAKER_ROLE_ARN).
        self.write_manifest(
            "svc",
            "sdk",
            "keys:\n  - API_KEY\n  - {name: ROLE_ARN, optional: true}\nprompt: hi\n",
        )
        self.assertEqual(interfaces.optional_keys("svc", "sdk"), {"ROLE_ARN"})
        self.assertEqual(interfaces.keys_for("svc", "sdk"), {"API_KEY": "", "ROLE_ARN": ""})
        # API_KEY present in env, ROLE_ARN absent → nothing missing.
        self.assertEqual(
            interfaces.missing_keys("svc", "sdk", env={"API_KEY": "x"}), []
        )
        # API_KEY absent → only the required key is reported.
        self.assertEqual(interfaces.missing_keys("svc", "sdk", env={}), ["API_KEY"])

    def test_optional_key_still_injected_when_present(self):
        self.write_manifest(
            "svc", "sdk", "keys:\n  - {name: ROLE_ARN, optional: true}\nprompt: hi\n"
        )
        got = interfaces._resolved_keys("svc", "sdk", env={"ROLE_ARN": "arn:aws:iam::1:role/x"})
        self.assertEqual(got, {"ROLE_ARN": "arn:aws:iam::1:role/x"})


class GraderInstallTests(InterfaceTestBase):
    """install_for_grader prefers the SDK manifest's `grader_install` (a thin
    read client) over its `runtime_install` (the full SDK), so CLI/MCP grading
    doesn't drag in the whole SDK just to import a thin client like boto3."""

    def _captured_steps(self, sdk_manifest: str) -> list:
        self.write_manifest("svc", "sdk", sdk_manifest)
        run_dir = self.root / "run"
        run_dir.mkdir()
        with mock.patch.object(interfaces, "_run_install") as ri:
            interfaces.install_for_grader("svc", run_dir, run_dir / "venv" / "bin" / "python")
        return list(ri.call_args[0][0]) if ri.called else []

    def test_grader_install_preferred_over_runtime_install(self):
        steps = self._captured_steps(
            "runtime_install:\n  - pip install bigsdk\n"
            "grader_install:\n  - pip install thinclient\nprompt: hi\n"
        )
        self.assertEqual(steps, ["pip install thinclient"])

    def test_falls_back_to_runtime_install_when_no_grader_install(self):
        steps = self._captured_steps("runtime_install:\n  - pip install bigsdk\nprompt: hi\n")
        self.assertEqual(steps, ["pip install bigsdk"])

    def test_noop_when_neither_declared(self):
        self.write_manifest("svc", "sdk", "prompt: hi\n")
        run_dir = self.root / "run2"
        run_dir.mkdir()
        with mock.patch.object(interfaces, "_run_install") as ri:
            interfaces.install_for_grader("svc", run_dir, run_dir / "venv" / "bin" / "python")
        self.assertFalse(ri.called)


class AccountingFieldsTests(InterfaceTestBase):
    """cli_command / sdk_module drive cli_calls / sdk_calls accounting."""

    def test_resolved_config_surfaces_overrides(self):
        self.write_manifest(
            "svc",
            "cli",
            "binary: svc_cli-0.1.0-py3-none-any.whl\ncli_command: svc\nprompt: hi\n",
        )
        cfg = interfaces._resolved_config("svc", "cli")
        self.assertEqual(cfg["cli_command"], "svc")
        self.assertEqual(cfg["binary"], "svc_cli-0.1.0-py3-none-any.whl")

    def test_resolved_config_defaults_none_when_absent(self):
        self.write_manifest("svc", "cli", "binary: hops\nprompt: hi\n")
        cfg = interfaces._resolved_config("svc", "cli")
        self.assertIsNone(cfg["cli_command"])  # falls back to binary in setup()
        self.assertIsNone(cfg["sdk_module"])
        self.assertEqual(cfg["teardown"], [])  # default: nothing to tear down

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
            interfaces._interface_dist_name({}, "hopsworks", "hopsworks-0-py3-none-any.whl"),
            "hopsworks",
        )

    def test_dist_name_falls_back_to_sdk_module_then_platform(self):
        self.assertEqual(
            interfaces._interface_dist_name({"sdk_module": "hsfs"}, "hopsworks", None),
            "hsfs",
        )
        self.assertEqual(interfaces._interface_dist_name({}, "hopsworks", None), "hopsworks")

    def test_ensure_base_clean_noop_for_none(self):
        # No-op for the null interface — and must not shell out to pip.
        import unittest.mock as mock

        with mock.patch.object(interfaces.subprocess, "run") as run:
            interfaces.ensure_base_clean("none", "none")
        run.assert_not_called()

    def test_ensure_base_clean_uninstalls_when_present(self):
        import unittest.mock as mock

        self.write_manifest("svc", "sdk", "binary: svc_pkg-0-py3-none-any.whl\nprompt: hi\n")
        calls = []

        def fake_run(cmd, *a, **k):
            calls.append(cmd)
            # Emulate `pip show <dist>` → present (exit 0); uninstall → exit 0.
            return mock.Mock(returncode=0)

        with mock.patch.object(interfaces.subprocess, "run", side_effect=fake_run):
            interfaces.ensure_base_clean("svc", "sdk")

        # A `pip show svc_pkg` probe followed by a `pip uninstall -y svc_pkg`.
        self.assertTrue(any("show" in c and "svc_pkg" in c for c in calls))
        self.assertTrue(any("uninstall" in c and "-y" in c and "svc_pkg" in c for c in calls))

    def test_ensure_base_clean_skips_uninstall_when_absent(self):
        import unittest.mock as mock

        self.write_manifest("svc", "sdk", "binary: svc_pkg-0-py3-none-any.whl\nprompt: hi\n")
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
            "svc",
            "sdk",
            "prompt: base\nallowed_domains: [api.svc.com]\ninstance_allowlist: [ml.m5.large]\n",
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
        self.write_manifest("svc", "sdk", 'keys:\n  API_KEY: ""\nprompt: hi\n')
        st = interfaces.preflight("svc", "sdk", env={})
        self.assertFalse(st.ok)
        self.assertIn("API_KEY", st.missing_keys)
        self.assertIn("setup", st.fix_command)

    def test_sdk_keys_present_passes(self):
        self.write_manifest("svc", "sdk", 'keys:\n  API_KEY: ""\nprompt: hi\n')
        st = interfaces.preflight("svc", "sdk", env={"API_KEY": "x"})
        self.assertTrue(st.ok)
        self.assertTrue(st.authenticated)

    def test_auth_command_success(self):
        self.write_manifest("svc", "cli", 'auth_command: "true"\nprompt: hi\n')
        st = interfaces.preflight("svc", "cli", env={})
        self.assertTrue(st.ok)

    def test_auth_command_failure(self):
        self.write_manifest("svc", "cli", 'auth_command: "false"\nprompt: hi\n')
        st = interfaces.preflight("svc", "cli", env={})
        self.assertFalse(st.ok)
        self.assertFalse(st.authenticated)

    def test_test_command_failure(self):
        self.write_manifest("svc", "cli", 'test_command: "false"\nprompt: hi\n')
        st = interfaces.preflight("svc", "cli", env={})
        self.assertFalse(st.ok)

    def test_missing_binary_without_build(self):
        self.write_manifest(
            "svc",
            "cli",
            "binary: tool\nruntime_install:\n  - cp $INTERFACE_DIR/tool .\nprompt: hi\n",
        )
        st = interfaces.preflight("svc", "cli", auto_build=False, env={})
        self.assertFalse(st.ok)
        self.assertFalse(st.installed)


class CheckAvailabilityTests(InterfaceTestBase):
    """Fast session-start gate: config present + creds present + skill bundle
    well-formed, with no build and no network."""

    def test_none_requirement_passes(self):
        preflight.check_availability([preflight.Requirement(platform="none", interface="none")])

    def test_missing_config_is_reported(self):
        with self.assertRaises(preflight.PreflightError) as cm:
            preflight.check_availability([preflight.Requirement(platform="ghost", interface="cli")])
        self.assertIn("no config manifest", str(cm.exception))

    def test_missing_credentials_reported(self):
        self.write_manifest("svc", "sdk", 'keys:\n  API_KEY: ""\nprompt: hi\n')
        with self.assertRaises(preflight.PreflightError) as cm:
            preflight.check_availability(
                [preflight.Requirement(platform="svc", interface="sdk")], env={}
            )
        self.assertIn("API_KEY", str(cm.exception))

    def test_credentials_present_passes(self):
        self.write_manifest("svc", "sdk", 'keys:\n  API_KEY: ""\nprompt: hi\n')
        preflight.check_availability(
            [preflight.Requirement(platform="svc", interface="sdk")], env={"API_KEY": "x"}
        )

    def test_all_problems_collected_in_one_raise(self):
        self.write_manifest("svc", "sdk", 'keys:\n  API_KEY: ""\nprompt: hi\n')
        with self.assertRaises(preflight.PreflightError) as cm:
            preflight.check_availability(
                [
                    preflight.Requirement(platform="svc", interface="sdk"),  # missing key
                    preflight.Requirement(platform="ghost", interface="cli"),  # no config
                ],
                env={},
            )
        msg = str(cm.exception)
        self.assertIn("API_KEY", msg)
        self.assertIn("no config manifest", msg)


class PreflightModuleTests(InterfaceTestBase):
    def test_none_requirement_passes(self):
        preflight.preflight(
            [preflight.Requirement(platform="none", interface="none")],
            auth="api-key",
            model="claude-sonnet-4-6",
            probe_skills=False,
        )

    def test_missing_interface_raises(self):
        with self.assertRaises(preflight.PreflightError):
            preflight.preflight(
                [preflight.Requirement(platform="ghost", interface="cli")],
                auth="api-key",
                model="claude-sonnet-4-6",
                probe_skills=False,
            )


class PrepareTests(InterfaceTestBase):
    """interfaces.prepare materializes a per-interface venv ONCE (hash-stamped),
    so runs clone it read-only instead of pip-installing per run."""

    def _fake_materialize(self, target: Path):
        # Stand in for the real venv build (no `python -m venv`, no base clone).
        (target / "bin").mkdir(parents=True, exist_ok=True)
        (target / "bin" / "python").write_text("#!/bin/sh\n")
        return target / "bin" / "python"

    def test_prepare_none_returns_none(self):
        self.assertIsNone(interfaces.prepare("none", "none"))

    def test_prepare_materializes_once_and_is_idempotent(self):
        self.write_manifest(
            "svc",
            "sdk",
            "runtime_install:\n  - pip install thing==1.0\nprompt: hi\n",
        )
        installs = []
        with (
            mock.patch.object(interfaces, "_materialize_venv", self._fake_materialize),
            mock.patch.object(
                interfaces, "_run_install", side_effect=lambda *a, **k: installs.append(a)
            ),
        ):
            venv = interfaces.prepare("svc", "sdk")
            self.assertTrue((venv / "bin" / "python").exists())
            self.assertEqual(venv, interfaces.prepared_venv_dir("svc", "sdk"))
            stamp = interfaces.bin_dir("svc", "sdk") / interfaces._PREPARED_STAMP
            self.assertTrue(stamp.exists())
            self.assertEqual(len(installs), 1)

            # Same manifest → same hash → reuse, no second install.
            interfaces.prepare("svc", "sdk")
            self.assertEqual(len(installs), 1)

            # Manifest change → hash change → rebuild + re-install.
            self.write_manifest(
                "svc",
                "sdk",
                "runtime_install:\n  - pip install thing==2.0\nprompt: hi\n",
            )
            interfaces.prepare("svc", "sdk")
            self.assertEqual(len(installs), 2)

    def test_base_venv_change_invalidates_prepared(self):
        # The stamp folds in a base-venv fingerprint, so a base change (new/
        # upgraded package, python bump) forces a rebuild — runs must not clone
        # a prepared venv built on a stale base.
        self.write_manifest(
            "svc", "sdk", "runtime_install:\n  - pip install thing==1.0\nprompt: hi\n"
        )
        installs = []
        with (
            mock.patch.object(interfaces, "_materialize_venv", self._fake_materialize),
            mock.patch.object(
                interfaces, "_run_install", side_effect=lambda *a, **k: installs.append(a)
            ),
            mock.patch.object(interfaces, "_base_venv_fingerprint", return_value="aaaa"),
        ):
            interfaces.prepare("svc", "sdk")
            interfaces.prepare("svc", "sdk")  # same base → reuse
            self.assertEqual(len(installs), 1)
        with (
            mock.patch.object(interfaces, "_materialize_venv", self._fake_materialize),
            mock.patch.object(
                interfaces, "_run_install", side_effect=lambda *a, **k: installs.append(a)
            ),
            mock.patch.object(interfaces, "_base_venv_fingerprint", return_value="bbbb"),
        ):
            interfaces.prepare("svc", "sdk")  # base changed → rebuild
            self.assertEqual(len(installs), 2)

    def test_venv_site_packages_pins_to_running_version(self):
        # A venv carrying a STALE tree from another python must resolve to the
        # running interpreter's tree, not an arbitrary glob pick (ABI safety).
        import sys

        venv = self.root / "v"
        cur = f"python{sys.version_info.major}.{sys.version_info.minor}"
        (venv / "lib" / cur / "site-packages").mkdir(parents=True)
        (venv / "lib" / "python2.7" / "site-packages").mkdir(parents=True)  # stale
        self.assertEqual(interfaces.venv_site_packages(venv), venv / "lib" / cur / "site-packages")

    def test_clean_build_artifacts_removes_prepared_venv(self):
        self.write_manifest("svc", "sdk", "prompt: hi\n")
        venv = interfaces.prepared_venv_dir("svc", "sdk")
        (venv / "bin").mkdir(parents=True, exist_ok=True)
        (venv / "bin" / "python").write_text("x")
        stamp = interfaces.bin_dir("svc", "sdk") / interfaces._PREPARED_STAMP
        stamp.write_text("abc")
        interfaces._clean_build_artifacts("svc", "sdk")
        self.assertFalse(venv.exists())
        self.assertFalse(stamp.exists())


if __name__ == "__main__":
    unittest.main()
