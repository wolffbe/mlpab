"""Unit tests for read-back retry/degrade in grade_table_main.

A client-side read-back flake (hsfs masks the real Arrow Flight error as a
generic FeatureStoreException) must NOT crash the grader — it should retry the
read (the server serves the rows reliably on repeat) and, only if it still
fails, degrade to an empty frame with the reason recorded. LookupError /
NotImplementedError are deterministic adapter limits and are not retried.
"""

import unittest
from unittest import mock

import pandas as pd

from evals import common


class FetchRetryTests(unittest.TestCase):
    def test_retries_then_succeeds(self):
        df = pd.DataFrame({"a": [1, 2]})
        calls = []

        def flaky(adapter, table, version, record_ids):
            calls.append(1)
            if len(calls) < 2:
                raise RuntimeError("Could not read data using Hopsworks Query Service.")
            return df

        with (
            mock.patch.object(common, "fetch_table", side_effect=flaky),
            mock.patch("time.sleep"),
        ):
            out = common.fetch_table_with_retry("hopsworks", "t", 1, None)
        self.assertEqual(len(calls), 2)
        self.assertTrue(out.equals(df))

    def test_gives_up_after_max_retries(self):
        with (
            mock.patch.object(common, "fetch_table", side_effect=RuntimeError("flake")),
            mock.patch("time.sleep") as slept,
        ):
            with self.assertRaises(RuntimeError):
                common.fetch_table_with_retry("hopsworks", "t", 1, None)
        # One sleep between each of the FETCH_RETRIES attempts (not after the last).
        self.assertEqual(slept.call_count, common.FETCH_RETRIES - 1)

    def test_does_not_retry_deterministic_errors(self):
        calls = []

        def miss(*a):
            calls.append(1)
            raise LookupError("no checker adapter")

        with (
            mock.patch.object(common, "fetch_table", side_effect=miss),
            mock.patch("time.sleep") as slept,
        ):
            with self.assertRaises(LookupError):
                common.fetch_table_with_retry("hopsworks", "t", 1, None)
        self.assertEqual(len(calls), 1)
        slept.assert_not_called()


if __name__ == "__main__":
    unittest.main()
