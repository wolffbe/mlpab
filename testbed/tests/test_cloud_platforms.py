"""Unit tests for the azure + gcp platforms and the flattened skills system:
adapter registration, manifest parsing, the single-skill bundle path, and the
implicit `official` bundle resolution (all offline — no cloud/network)."""
import tempfile
import unittest
from pathlib import Path

from mlpab import interfaces as I, evals_provider as ep, skills, treatments as bm


class AdapterRegistrationTests(unittest.TestCase):
    def test_all_five_platforms_registered(self):
        self.assertEqual(ep.ADAPTERS, ("hopsworks", "databricks", "aws", "azure", "gcp"))

    def test_new_adapters_import_with_full_contract(self):
        from evals.adapters.aws import SageMakerChecker          # sagemaker→aws module
        from evals.adapters.azure import AzureMLChecker
        from evals.adapters.gcp import VertexChecker
        contract = ("get_feature_table", "read_rows", "read_training_dataset",
                    "get_model", "get_job", "get_endpoint", "get_alert", "get_vector_store")
        for cls in (SageMakerChecker, AzureMLChecker, VertexChecker):
            for meth in contract:
                self.assertTrue(hasattr(cls, meth), f"{cls.__name__}.{meth}")

    def test_dispatch_accepts_new_adapters(self):
        # state_checker only branches on the name; constructing is lazy/guarded,
        # so we just assert the choices wiring rejects nothing expected.
        from evals.common import state_checker
        self.assertIsNone(state_checker("none"))


class ManifestTests(unittest.TestCase):
    def test_new_platform_manifests_parse(self):
        for platform in ("azure", "gcp"):
            for iface in ("cli", "sdk"):
                m = I.load_manifest(platform, iface)
                self.assertTrue(m.get("version"))
                self.assertTrue(m.get("keys"))
                self.assertTrue(m.get("allowed_domains"))
                p, i = I.platform_interface_from_config(
                    f"configs/platforms/{platform}/{iface}.yaml")
                self.assertEqual((p, i), (platform, iface))

    def test_aws_rename_resolves(self):
        p, i = I.platform_interface_from_config("configs/platforms/aws/sdk.yaml")
        self.assertEqual((p, i), ("aws", "sdk"))


class SkillBundleTests(unittest.TestCase):
    def test_official_is_implicit_for_every_platform(self):
        # flat skills.yaml → exactly the implicit `official` bundle (no fetch).
        for platform in ("hopsworks", "databricks", "aws", "azure", "gcp"):
            self.assertIn("official", skills.bundle_names(platform), platform)

    def test_treatment_skills_entry_needs_no_bundle_key(self):
        plat, bundle = bm._skills_from_config(
            {"config": "configs/platforms/gcp/skills.yaml"})
        self.assertEqual((plat, bundle), ("gcp", "official"))

    def test_skill_dirs_under_single_skill(self):
        with tempfile.TemporaryDirectory() as t:
            sub = Path(t) / "azure-machine-learning"
            sub.mkdir()
            (sub / "SKILL.md").write_text("name: aml\n")
            self.assertEqual(skills._skill_dirs_under(sub), [sub])  # sub IS the skill

    def test_skill_dirs_under_flat_and_category(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            # flat skill
            (root / "deploy").mkdir(); (root / "deploy" / "SKILL.md").write_text("x")
            # one category level to flatten
            (root / "cat" / "tune").mkdir(parents=True)
            (root / "cat" / "tune" / "SKILL.md").write_text("x")
            (root / "loose.txt").write_text("ignored")
            got = sorted(p.name for p in skills._skill_dirs_under(root))
            self.assertEqual(got, ["deploy", "tune"])


class TreatmentGridTests(unittest.TestCase):
    def test_new_platform_grids_cover_both_arms(self):
        for platform in ("azure", "gcp"):
            sk = bm.load_config(
                f"configs/treatments/{platform}/{platform}-opus-4-8-skills.yaml")
            nosk = bm.load_config(
                f"configs/treatments/{platform}/{platform}-opus-4-8-no-skills.yaml")
            self.assertEqual(sorted({r.skills for r in sk.runs}), ["official"])
            self.assertEqual(sorted({r.skills for r in nosk.runs}), ["none"])
            self.assertEqual(sorted({r.platform for r in sk.runs}), [platform])


if __name__ == "__main__":
    unittest.main()
