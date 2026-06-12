"""Tests for runner helpers: dynamic environment detection + prompt rendering."""
import os
import unittest
from unittest import mock

from banter import evals_provider, runner


class SeedForTests(unittest.TestCase):
    def test_deterministic_and_distinct_per_attempt(self):
        a = evals_provider.seed_for("rq1", "feature", "training_data", 1)
        self.assertEqual(a, evals_provider.seed_for("rq1", "feature", "training_data", 1))
        self.assertNotEqual(a, evals_provider.seed_for("rq1", "feature", "training_data", 2))
        # Non-negative 31-bit int (usable as a numpy/random seed).
        self.assertGreaterEqual(a, 0)
        self.assertLess(a, 2 ** 31)

    def test_unimplemented_family_fails_fast(self):
        with self.assertRaises(ValueError):
            evals_provider._family("not-a-real-eval")


class PlatformEnvTests(unittest.TestCase):
    """_platform_env: the KEY=VALUE handoff file a platform setup step writes
    (via $BANTER_PLATFORM_ENV) so values it derives — e.g. the SageMaker
    execution-role ARN it just created — reach the agent's env."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        self.run_dir = Path(tempfile.mkdtemp())

    def test_missing_file_is_empty(self):
        self.assertEqual(runner._platform_env(self.run_dir), {})

    def test_parses_and_consumes_file(self):
        path = self.run_dir / "platform.env"
        path.write_text(
            "# from setup.py\n"
            "SAGEMAKER_ROLE_ARN=arn:aws:iam::123:role/banter-sagemaker-execution-role\n"
            "\n"
            "malformed line without equals\n"
            "SPACED = padded value \n"
        )
        env = runner._platform_env(self.run_dir)
        self.assertEqual(env, {
            "SAGEMAKER_ROLE_ARN": "arn:aws:iam::123:role/banter-sagemaker-execution-role",
            "SPACED": "padded value",
        })
        # Consumed: deleted so it never lingers in the agent's workdir.
        self.assertFalse(path.exists())

    def test_declared_keys_win_on_merge(self):
        (self.run_dir / "platform.env").write_text("SAGEMAKER_ROLE_ARN=derived\nEXTRA=kept\n")
        keys = {"SAGEMAKER_ROLE_ARN": "from-dotenv"}
        merged = {**runner._platform_env(self.run_dir), **keys}
        self.assertEqual(merged, {"SAGEMAKER_ROLE_ARN": "from-dotenv", "EXTRA": "kept"})


class RunAuxTests(unittest.TestCase):
    """Platform setup/teardown subprocesses must not be instrumented: their REST
    traffic is plumbing, not agent work, so the api-log env vars are stripped
    before the step runs (the per-run venv's shim logs whenever BANTER_API_LOG
    is set, and these steps inherit os.environ wholesale)."""

    def test_strips_api_log_instrumentation_env(self):
        import tempfile
        from pathlib import Path

        run_dir = Path(tempfile.mkdtemp())
        out = run_dir / "env.txt"
        env = {
            "PATH": os.environ.get("PATH", ""),
            "BANTER_API_LOG": str(run_dir / "api_calls.jsonl"),
            "BANTER_IFACE_SDK": "hopsworks",
            "HOPSWORKS_API_KEY": "kept",
        }
        runner._run_aux([f'env > "{out}"'], run_dir, env)
        text = out.read_text()
        self.assertNotIn("BANTER_API_LOG", text)
        self.assertNotIn("BANTER_IFACE_SDK", text)
        self.assertIn("HOPSWORKS_API_KEY=kept", text)
        # The caller's env dict is not mutated.
        self.assertIn("BANTER_API_LOG", env)

class DetectEnvironmentTests(unittest.TestCase):
    def test_includes_cores_and_start_method(self):
        text = runner.detect_environment()
        self.assertIn("CPU cores", text)
        self.assertIn("start method", text)

    def test_spawn_platform_warns_about_dataloader(self):
        with mock.patch("platform.system", return_value="Darwin"), \
             mock.patch("platform.machine", return_value="arm64"), \
             mock.patch("shutil.which", return_value=None):
            text = runner.detect_environment()
        self.assertIn("'spawn'", text)
        self.assertIn("DEADLOCK", text)
        self.assertIn("num_workers", text)
        self.assertIn("MPS", text)

    def test_fork_platform_omits_deadlock_warning(self):
        with mock.patch("platform.system", return_value="Linux"), \
             mock.patch("platform.machine", return_value="x86_64"), \
             mock.patch("shutil.which", return_value=None):
            text = runner.detect_environment()
        self.assertIn("'fork'", text)
        self.assertNotIn("DEADLOCK", text)
        self.assertIn("No CUDA GPU", text)

    def test_nvidia_present_mentions_cuda(self):
        with mock.patch("platform.system", return_value="Linux"), \
             mock.patch("platform.machine", return_value="x86_64"), \
             mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            text = runner.detect_environment()
        self.assertIn("CUDA", text)


class BuildPromptTests(unittest.TestCase):
    def test_fills_all_placeholders(self):
        prompt = runner._build_prompt("TASK BODY TEXT", "INTERFACE_FRAGMENT")
        self.assertIn("TASK BODY TEXT", prompt)
        self.assertIn("INTERFACE_FRAGMENT", prompt)
        # The {environment} placeholder is resolved (detected bullets present).
        self.assertIn("CPU cores", prompt)
        # No unfilled placeholders leaked through.
        self.assertNotIn("{environment}", prompt)
        self.assertNotIn("{task_body}", prompt)
        self.assertNotIn("{fragment}", prompt)

    def test_interface_under_test_section(self):
        # With an interface present (default), the under-test rule is included
        # and the HTML markers are stripped.
        on = runner._build_prompt("c", "FRAG", interface_under_test=True)
        self.assertIn("The interface is what's being measured", on)
        self.assertNotIn("UNDER_TEST_START", on)
        self.assertNotIn("UNDER_TEST_END", on)

        # For none/none the whole section is removed.
        off = runner._build_prompt("c", "FRAG", interface_under_test=False)
        self.assertNotIn("The interface is what's being measured", off)
        self.assertNotIn("UNDER_TEST", off)
        self.assertNotIn("\n\n\n", off)  # no collapsed-newline gap left behind


if __name__ == "__main__":
    unittest.main()
