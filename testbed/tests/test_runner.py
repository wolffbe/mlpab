"""Tests for runner helpers: dynamic environment detection + prompt rendering."""
import unittest
from unittest import mock

from banter import mlebench_wrapper, runner


class NormalizeScoreTests(unittest.TestCase):
    def test_higher_better_passes_through(self):
        # AUC / accuracy: already higher-is-better.
        self.assertEqual(mlebench_wrapper.normalize_score(0.93, False), 0.93)
        self.assertEqual(mlebench_wrapper.normalize_score(0.93, 0), 0.93)

    def test_lower_better_is_sign_flipped(self):
        # RMSE / RMSLE: smaller error must map to a LARGER normalized value.
        self.assertEqual(mlebench_wrapper.normalize_score(0.05, True), -0.05)
        self.assertEqual(mlebench_wrapper.normalize_score(3.0, 1), -3.0)
        # A better (smaller) error normalizes higher than a worse (larger) one.
        self.assertGreater(
            mlebench_wrapper.normalize_score(0.04, True),
            mlebench_wrapper.normalize_score(0.05, True),
        )

    def test_none_passes_through(self):
        self.assertIsNone(mlebench_wrapper.normalize_score(None, True))
        self.assertIsNone(mlebench_wrapper.normalize_score(None, False))


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
        prompt = runner._build_prompt("my-challenge", "INTERFACE_FRAGMENT")
        self.assertIn("my-challenge", prompt)
        self.assertIn("INTERFACE_FRAGMENT", prompt)
        # The {environment} placeholder is resolved (detected bullets present).
        self.assertIn("CPU cores", prompt)
        # No unfilled placeholders leaked through.
        self.assertNotIn("{environment}", prompt)
        self.assertNotIn("{challenge_id}", prompt)
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
