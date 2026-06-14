"""Unit tests for secret redaction (redact.py)."""

import unittest

from mlpab import redact


class RedactTests(unittest.TestCase):
    def test_redacts_secret_named_long_values(self):
        env = {"DATABRICKS_TOKEN": "dapiSECRET1234567890", "MISTRAL_API_KEY": "msk-abcdefghij"}
        out = redact.redact("tok=dapiSECRET1234567890 key=msk-abcdefghij", env=env)
        self.assertNotIn("dapiSECRET1234567890", out)
        self.assertNotIn("msk-abcdefghij", out)
        self.assertEqual(out.count(redact.PLACEHOLDER), 2)

    def test_keeps_non_secret_and_short_values(self):
        env = {
            "AWS_REGION": "us-east-1",
            "API_KEY": "short",
        }  # region not secret-named; "short" too short
        out = redact.redact("region=us-east-1 k=short", env=env)
        self.assertEqual(out, "region=us-east-1 k=short")

    def test_extra_values_are_redacted(self):
        out = redact.redact("inline=ABCDEFGH123", extra=["ABCDEFGH123"], env={})
        self.assertIn(redact.PLACEHOLDER, out)
        self.assertNotIn("ABCDEFGH123", out)

    def test_longest_first_no_partial_leak(self):
        env = {"X_TOKEN": "abcdefgh", "Y_SECRET": "abcdefgh-extra-tail"}
        out = redact.redact("v=abcdefgh-extra-tail", env=env)
        # The longer secret must be fully masked, not leave "-extra-tail" behind.
        self.assertNotIn("abcdefgh-extra-tail", out)
        self.assertNotIn("-extra-tail", out)

    def test_empty_and_none(self):
        self.assertEqual(redact.redact("", env={}), "")
        self.assertEqual(redact.redact(None, env={}), "")


if __name__ == "__main__":
    unittest.main()
