"""Tests for runner helpers: dynamic environment detection + prompt rendering."""
import unittest
from unittest import mock

from banter import runner


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


if __name__ == "__main__":
    unittest.main()
