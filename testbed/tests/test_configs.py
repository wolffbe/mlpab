"""Tests for config parsing: platform config references + unified tasks."""
import tempfile
import unittest
from pathlib import Path

from banter import autoresearch as ar, benchmark as bm, interfaces


class PlatformInterfaceFromConfigTests(unittest.TestCase):
    def test_derives_platform_and_interface(self):
        self.assertEqual(
            interfaces.platform_interface_from_config("platforms/hopsworks/cli/config.yaml"),
            ("hopsworks", "cli"),
        )

    def test_unknown_interface_raises(self):
        with self.assertRaises(ValueError):
            interfaces.platform_interface_from_config("platforms/hopsworks/bogus/config.yaml")


def _write(text: str) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(text)
    f.close()
    return Path(f.name)


class AutoresearchConfigTests(unittest.TestCase):
    def test_tasks_and_platform_config_refs(self):
        p = _write(
            """
improve: [interface]
skills: none
tasks:
  image_classification: [aerial-cactus-identification]
  tabular: [nomad2018-predict-transparent-conductors, tabular-playground-series-dec-2021]
interfaces:
  - config: platforms/hopsworks/cli/config.yaml
  - config: platforms/hopsworks/mcp/config.yaml
goals: [score]
budget: {max_increments: 3}
"""
        )
        cfg = ar.load_config(p)
        self.assertEqual(list(cfg.tasks), ["image_classification", "tabular"])
        # challenges flattened (deduped, order-preserving)
        self.assertEqual(len(cfg.challenges), 3)
        self.assertEqual([(i.platform, i.interface) for i in cfg.interfaces],
                         [("hopsworks", "cli"), ("hopsworks", "mcp")])
        self.assertEqual(cfg.interfaces[0].config, "platforms/hopsworks/cli/config.yaml")
        self.assertEqual(cfg.skills, "none")
        self.assertEqual(cfg.budget.max_increments, 3)

    def test_legacy_keys_still_parse(self):
        p = _write(
            """
improve: interface
starting_skills: none
challenges: [aerial-cactus-identification]
starting_interfaces:
  - {platform: hopsworks, interface: cli}
goals: [score]
budget: {max_cycles: 2}
"""
        )
        cfg = ar.load_config(p)
        self.assertEqual(cfg.tasks, {"all": ["aerial-cactus-identification"]})
        self.assertEqual(cfg.interfaces[0].platform, "hopsworks")
        self.assertEqual(cfg.budget.max_increments, 2)

    def test_max_seconds_defaults_to_8h(self):
        p = _write(
            """
improve: interface
skills: none
challenges: [aerial-cactus-identification]
interfaces: [{platform: hopsworks, interface: cli}]
goals: [score]
budget: {max_increments: 3}
"""
        )
        cfg = ar.load_config(p)
        self.assertEqual(cfg.budget.max_seconds, 8 * 3600.0)

    def test_max_seconds_override(self):
        p = _write(
            """
improve: interface
skills: none
challenges: [aerial-cactus-identification]
interfaces: [{platform: hopsworks, interface: cli}]
goals: [score]
budget: {max_increments: 3, max_seconds: 3600}
"""
        )
        cfg = ar.load_config(p)
        self.assertEqual(cfg.budget.max_seconds, 3600.0)


class BenchmarkConfigTests(unittest.TestCase):
    def test_matrix_platform_config_ref(self):
        p = _write(
            """
engineer_model: claude-sonnet-4-6
challenges: [aerial-cactus-identification]
interfaces:
  - {config: platforms/hopsworks/cli/config.yaml, session: abc123, version: 2}
skills: [none]
"""
        )
        cfg = bm.load_config(p)
        self.assertEqual(len(cfg.runs), 1)
        r = cfg.runs[0]
        self.assertEqual((r.platform, r.interface, r.session, r.interface_version),
                         ("hopsworks", "cli", "abc123", 2))


if __name__ == "__main__":
    unittest.main()
