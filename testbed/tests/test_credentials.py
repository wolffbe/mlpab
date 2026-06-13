"""Unit tests for agent-sandbox credential localization: file-path creds
(azure cert / gcp ADC) are copied into the run dir and repointed; value-based
creds and missing files are left untouched; the input dict is not mutated."""
import tempfile
import unittest
from pathlib import Path

from mlpab.runner import _localize_file_credentials


class LocalizeFileCredentialsTests(unittest.TestCase):
    def test_copies_file_cred_into_run_dir_and_repoints(self):
        d = Path(tempfile.mkdtemp())
        cert = d / "host" / "azure-sp.pem"
        cert.parent.mkdir(); cert.write_text("PRIVATE KEY")
        run_dir = d / "run"; run_dir.mkdir()
        keys = {"AZURE_CLIENT_CERTIFICATE_PATH": str(cert),
                "DATABRICKS_TOKEN": "dapi-secret"}
        out = _localize_file_credentials(keys, run_dir)
        new = Path(out["AZURE_CLIENT_CERTIFICATE_PATH"])
        self.assertTrue(new.is_file())
        self.assertTrue(str(new).startswith(str(run_dir)))      # inside the sandbox
        self.assertEqual(new.read_text(), "PRIVATE KEY")
        self.assertEqual(out["DATABRICKS_TOKEN"], "dapi-secret")  # value cred untouched

    def test_missing_file_and_value_creds_unchanged(self):
        run_dir = Path(tempfile.mkdtemp())
        keys = {"GOOGLE_APPLICATION_CREDENTIALS": "/no/such/file.json",
                "AWS_SECRET_ACCESS_KEY": "xyz"}
        out = _localize_file_credentials(keys, run_dir)
        self.assertEqual(out["GOOGLE_APPLICATION_CREDENTIALS"], "/no/such/file.json")
        self.assertEqual(out["AWS_SECRET_ACCESS_KEY"], "xyz")
        self.assertFalse((run_dir / ".creds").exists())

    def test_does_not_mutate_input(self):
        d = Path(tempfile.mkdtemp())
        adc = d / "adc.json"; adc.write_text("{}")
        run_dir = d / "run"; run_dir.mkdir()
        keys = {"GOOGLE_APPLICATION_CREDENTIALS": str(adc)}
        _localize_file_credentials(keys, run_dir)
        self.assertEqual(keys["GOOGLE_APPLICATION_CREDENTIALS"], str(adc))  # original kept


if __name__ == "__main__":
    unittest.main()
