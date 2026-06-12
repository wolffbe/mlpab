"""Tests for config parsing: platform config references + unified tasks."""
import tempfile
import unittest
from pathlib import Path

from banter import treatments as bm, interfaces


class PlatformInterfaceFromConfigTests(unittest.TestCase):
    def test_derives_platform_and_interface(self):
        self.assertEqual(
            interfaces.platform_interface_from_config("configs/platforms/hopsworks/cli.yaml"),
            ("hopsworks", "cli"),
        )

    def test_unknown_interface_raises(self):
        with self.assertRaises(ValueError):
            interfaces.platform_interface_from_config("configs/platforms/hopsworks/bogus.yaml")


def _write(text: str) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(text)
    f.close()
    return Path(f.name)


class TreatmentRepeatsTests(unittest.TestCase):
    def test_repeats_default_one(self):
        p = _write("tasks: {feature: [t1]}\ninterfaces: [{platform: none, interface: none}]\n")
        self.assertEqual(bm.load_config(p).repeats, 1)

    def test_n_key_sets_repeats(self):
        p = _write("n: 3\ntasks: {feature: [t1]}\ninterfaces: [{platform: none, interface: none}]\n")
        self.assertEqual(bm.load_config(p).repeats, 3)

    def test_repeats_key_alias(self):
        p = _write("repeats: 2\ntasks: {feature: [t1]}\ninterfaces: [{platform: none, interface: none}]\n")
        self.assertEqual(bm.load_config(p).repeats, 2)


class TreatmentConfigTests(unittest.TestCase):
    def test_matrix_platform_config_ref(self):
        p = _write(
            """
model: claude-sonnet-4-6
tasks:
  feature: [training_data]
interfaces:
  - {config: configs/platforms/hopsworks/cli.yaml}
skills: [none]
"""
        )
        cfg = bm.load_config(p)
        self.assertEqual(len(cfg.runs), 1)
        r = cfg.runs[0]
        self.assertEqual((r.platform, r.interface), ("hopsworks", "cli"))
        self.assertEqual(cfg.model, "claude-sonnet-4-6")

    def test_tasks_mapping_groups_tasks_by_category(self):
        p = _write(
            """
tasks:
  feature: [training_data]
  inference: [batch_scoring]
interfaces:
  - {platform: hopsworks, interface: cli}
skills: [none]
"""
        )
        cfg = bm.load_config(p)
        self.assertEqual([(r.category, r.task) for r in cfg.runs],
                         [("feature", "training_data"), ("inference", "batch_scoring")])

    def test_runs_entries_use_task_and_category_keys(self):
        p = _write(
            """
runs:
  - task: training_data
    category: feature
    platform: none
    interface: none
"""
        )
        r = bm.load_config(p).runs[0]
        self.assertEqual((r.category, r.task), ("feature", "training_data"))


if __name__ == "__main__":
    unittest.main()
