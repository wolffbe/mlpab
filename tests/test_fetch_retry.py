"""Unit tests for the single-attempt deliverable read in the graders.

The Hopsworks Query Service is reliable (live-probed: a healthy feature group
reads back without flaking), so a grader read is attempted ONCE and NOT retried.
A read failure means the deliverable could not be read back — on a healthy
service that is "no results", which the grader degrades to (empty frame + reason)
rather than crashing. LookupError/NotImplementedError are deterministic adapter
limits / genuine misses; a registered-but-empty feature group is surfaced as a
clean LookupError.
"""

import unittest
from unittest import mock

import pandas as pd

from evals import common


class ReadDeliverableTests(unittest.TestCase):
    def test_success_returns_df_single_call(self):
        df = pd.DataFrame({"a": [1, 2]})
        calls = []

        def read(adapter, table, version, record_ids):
            calls.append(1)
            return df

        with mock.patch.object(common, "fetch_table", side_effect=read):
            out = common.fetch_table_deliverable("hopsworks", "t", 1, None)
        self.assertTrue(out.equals(df))
        self.assertEqual(len(calls), 1)

    def test_read_failure_raises_once_without_retry(self):
        # Not flaky: a read error is surfaced immediately (no retry loop, no
        # sleep), with the masked root cause appended for an actionable reason.
        calls = []

        def flaky(*a):
            calls.append(1)
            raise RuntimeError("Could not read data using Hopsworks Query Service.")

        with (
            mock.patch.object(common, "fetch_table", side_effect=flaky),
            mock.patch("time.sleep") as slept,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                common.fetch_table_deliverable("hopsworks", "t", 1, None)
        self.assertEqual(len(calls), 1)  # attempted exactly once
        slept.assert_not_called()
        self.assertIn("root cause", str(ctx.exception))

    def test_deterministic_errors_pass_through(self):
        calls = []

        def miss(*a):
            calls.append(1)
            raise LookupError("no checker adapter")

        with mock.patch.object(common, "fetch_table", side_effect=miss):
            with self.assertRaises(LookupError):
                common.fetch_table_deliverable("hopsworks", "t", 1, None)
        self.assertEqual(len(calls), 1)

    def test_empty_feature_group_becomes_clean_lookup_error(self):
        # The Arrow Flight server states an empty FG plainly; hsfs masks it behind
        # the generic FeatureStoreException. The cause chain carries the real
        # message, so it is surfaced as a clean LookupError ("no rows ingested").
        calls = []

        def empty(*a):
            calls.append(1)
            cause = RuntimeError("No active delta files found; no data has been written yet")
            raise RuntimeError("Could not read data using Hopsworks Query Service.") from cause

        with mock.patch.object(common, "fetch_table", side_effect=empty):
            with self.assertRaises(LookupError) as ctx:
                common.fetch_table_deliverable("hopsworks", "t", 1, None)
        self.assertIn("no rows were ingested", str(ctx.exception))
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
