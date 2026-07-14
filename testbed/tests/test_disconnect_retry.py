"""Unit tests for transport-disconnect detection and retry in claude_runner.

A socket drop between Claude Code and the Anthropic API ("socket connection was
closed unexpectedly") is an instantaneous network blip, not a rate-limit or a
capability failure. It must be RETRIED a small bounded number of times rather
than recorded as a dead, zeroed run. The detection phrases are specific to the
Node transport layer so platform output echoed in a tool result (e.g. a
server-side "Connection closed." log) never trips a retry.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mlpab import claude_runner as cr


class DisconnectDetectionTests(unittest.TestCase):
    def test_matches_node_socket_phrases(self):
        for txt in (
            "API Error: The socket connection was closed unexpectedly.",
            "FetchError: socket hang up",
            "read ECONNRESET",
            "Client network socket disconnected before secure TLS connection",
        ):
            self.assertTrue(cr.text_is_disconnect_error(txt), txt)

    def test_ignores_platform_connection_text(self):
        # A platform/SDK log echoed in a tool result must NOT count as a Claude
        # transport disconnect — these are the false positives we must avoid.
        for txt in (
            "2026-06-17 INFO: Connection closed.",
            "hopsworks_common.client: connection to the feature store closed",
            "Server response: HTTP code: 404, Not Found",
        ):
            self.assertFalse(cr.text_is_disconnect_error(txt), txt)

    def _write_result(self, path: Path, *, is_error: bool, text: str) -> None:
        path.write_text(
            json.dumps({"type": "result", "is_error": is_error, "subtype": "x", "result": text})
            + "\n"
        )

    def test_attempt_disconnect_from_result_blob(self):
        with TemporaryDirectory() as d:
            tr = Path(d) / "transcript.jsonl"
            err = Path(d) / "err"
            err.write_text("")
            self._write_result(tr, is_error=True, text="socket hang up")
            self.assertTrue(cr._attempt_is_disconnect(tr, err))

    def test_attempt_disconnect_from_stream_tail(self):
        # Stop reason was stop_sequence (result not flagged is_error), but the
        # disconnect printed as a streamed line — caught via the transcript tail.
        with TemporaryDirectory() as d:
            tr = Path(d) / "transcript.jsonl"
            err = Path(d) / "err"
            err.write_text("")
            tr.write_text(
                json.dumps({"type": "assistant", "message": {"content": []}})
                + "\n"
                + "API Error: The socket connection was closed unexpectedly.\n"
                + json.dumps({"type": "result", "is_error": False, "result": "ok"})
                + "\n"
            )
            self.assertTrue(cr._attempt_is_disconnect(tr, err))

    def test_attempt_no_disconnect_on_clean_failure(self):
        with TemporaryDirectory() as d:
            tr = Path(d) / "transcript.jsonl"
            err = Path(d) / "err"
            err.write_text("some unrelated stderr noise")
            self._write_result(tr, is_error=True, text="Server response: 404 Not Found")
            self.assertFalse(cr._attempt_is_disconnect(tr, err))


if __name__ == "__main__":
    unittest.main()
