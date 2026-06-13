"""Tests for the live stream-json terminal display (mlpab.streaming)."""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from mlpab import streaming


def _assistant(*blocks):
    return {"type": "assistant", "message": {"content": list(blocks)}}


class QuietTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("MLPAB_QUIET")
        os.environ.pop("MLPAB_QUIET", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("MLPAB_QUIET", None)
        else:
            os.environ["MLPAB_QUIET"] = self._saved

    def test_default_is_loud(self):
        self.assertFalse(streaming.quiet())

    def test_truthy_values_silence(self):
        for v in ("1", "true", "yes", "TRUE"):
            os.environ["MLPAB_QUIET"] = v
            self.assertTrue(streaming.quiet(), v)

    def test_falsey_values_stay_loud(self):
        for v in ("", "0", "false", "no"):
            os.environ["MLPAB_QUIET"] = v
            self.assertFalse(streaming.quiet(), v)

    def test_make_printer_always_callable(self):
        # Never None — stream.log must be written even when quiet/nested.
        os.environ["MLPAB_QUIET"] = "1"
        self.assertTrue(callable(streaming.make_printer("agent")))
        os.environ.pop("MLPAB_QUIET", None)
        self.assertTrue(callable(streaming.make_printer("agent")))


class NestedTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("MLPAB_NESTED", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["MLPAB_NESTED"] = self._saved
        else:
            os.environ.pop("MLPAB_NESTED", None)

    def test_default_not_nested(self):
        self.assertFalse(streaming.nested())

    def test_set_is_nested(self):
        os.environ["MLPAB_NESTED"] = "1"
        self.assertTrue(streaming.nested())


class EmitTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("MLPAB_QUIET", None)

    def test_writes_file_and_stdout(self):
        log = Path(tempfile.mkdtemp()) / "stream.log"
        buf = io.StringIO()
        with redirect_stdout(buf):
            streaming.emit("[eng] hi", log)
        self.assertEqual(buf.getvalue(), "[eng] hi\n")
        self.assertEqual(log.read_text(), "[eng] hi\n")

    def test_quiet_suppresses_stdout_but_still_writes_file(self):
        os.environ["MLPAB_QUIET"] = "1"
        log = Path(tempfile.mkdtemp()) / "stream.log"
        buf = io.StringIO()
        with redirect_stdout(buf):
            streaming.emit("[eng] hi", log)
        self.assertEqual(buf.getvalue(), "")
        self.assertEqual(log.read_text(), "[eng] hi\n")


class FileTailerTests(unittest.TestCase):
    def test_poll_prints_only_new_lines(self):
        root = Path(tempfile.mkdtemp())
        run = root / "inc0_combo" / "task" / "chal"
        run.mkdir(parents=True)
        log = run / "stream.log"
        log.write_text("[agent] a\n[agent:bash] ls\n")
        tailer = streaming.FileTailer(root, "**/stream.log")

        buf = io.StringIO()
        with redirect_stdout(buf):
            tailer.poll_once()
        self.assertEqual(buf.getvalue(), "[agent] a\n[agent:bash] ls\n")

        # Append; only the new line is printed on the next poll.
        with log.open("a") as f:
            f.write("[agent] b\n")
        buf = io.StringIO()
        with redirect_stdout(buf):
            tailer.poll_once()
        self.assertEqual(buf.getvalue(), "[agent] b\n")

    def test_excludes_paths(self):
        root = Path(tempfile.mkdtemp())
        controller_log = root / "stream.log"
        controller_log.write_text("[controller] x\n")
        tailer = streaming.FileTailer(root, "**/stream.log", exclude=(controller_log,))
        buf = io.StringIO()
        with redirect_stdout(buf):
            tailer.poll_once()
        self.assertEqual(buf.getvalue(), "")

    def test_truncated_file_re_read(self):
        root = Path(tempfile.mkdtemp())
        log = root / "stream.log"
        log.write_text("[agent] a longer original line\n")
        tailer = streaming.FileTailer(root, "**/stream.log")
        with redirect_stdout(io.StringIO()):
            tailer.poll_once()
        # Recreate smaller (combo re-run) → re-read from the start.
        log.write_text("[agent] new\n")
        buf = io.StringIO()
        with redirect_stdout(buf):
            tailer.poll_once()
        self.assertEqual(buf.getvalue(), "[agent] new\n")


class AssistantLinesTests(unittest.TestCase):
    def test_text_block_one_line_per_source_line(self):
        ev = _assistant({"type": "text", "text": "hello\n\nworld"})
        self.assertEqual(
            streaming.assistant_lines(ev, "agent"),
            ["[agent] hello", "[agent] world"],
        )

    def test_bash_shows_full_command_one_line_per_source_line(self):
        # New behaviour: live log surfaces the WHOLE bash command body, one
        # tagged terminal line per source line. The verbatim raw event still
        # lands in transcript.jsonl for tooling that wants the structured form.
        cmd = "echo first\nsecond\nthird"
        ev = _assistant({"type": "tool_use", "name": "Bash", "input": {"command": cmd}})
        self.assertEqual(
            streaming.assistant_lines(ev, "agent"),
            ["[agent:bash] echo first", "[agent:bash] second", "[agent:bash] third"],
        )

    def test_bash_does_not_truncate_long_lines(self):
        long = "x" * 500
        ev = _assistant({"type": "tool_use", "name": "Bash", "input": {"command": long}})
        (line,) = streaming.assistant_lines(ev, "agent")
        self.assertEqual(line, f"[agent:bash] {long}")

    def test_file_tools_show_path(self):
        for name in ("Read", "Write", "Edit"):
            ev = _assistant({"type": "tool_use", "name": name, "input": {"file_path": "/tmp/x.py"}})
            self.assertEqual(
                streaming.assistant_lines(ev, "eng"),
                [f"[eng:{name.lower()}] /tmp/x.py"],
            )

    def test_mcp_tool(self):
        ev = _assistant({"type": "tool_use", "name": "mcp__hops__list", "input": {}})
        self.assertEqual(streaming.assistant_lines(ev, "eng"), ["[eng:mcp] mcp__hops__list"])

    def test_other_tool_falls_back_to_name(self):
        ev = _assistant({"type": "tool_use", "name": "WebFetch", "input": {}})
        self.assertEqual(streaming.assistant_lines(ev, "eng"), ["[eng:tool] WebFetch"])

    def test_non_dict_blocks_ignored(self):
        ev = {"type": "assistant", "message": {"content": ["a string", None]}}
        self.assertEqual(streaming.assistant_lines(ev, "eng"), [])

    def test_thinking_block_shows_every_line(self):
        # "Show everything": each line of the model's reasoning is surfaced
        # on its own tagged terminal line — no first-line-only truncation.
        ev = _assistant({"type": "thinking", "thinking": "Let me plan this\nmore detail"})
        self.assertEqual(
            streaming.assistant_lines(ev, "eng"),
            ["[eng:thinking] Let me plan this", "[eng:thinking] more detail"],
        )


class ToolResultLinesTests(unittest.TestCase):
    def _user(self, *blocks):
        return {"type": "user", "message": {"content": list(blocks)}}

    def test_string_content_shows_all_lines(self):
        ev = self._user({"type": "tool_result", "content": "world\nsecond"})
        self.assertEqual(
            streaming.tool_result_lines(ev, "eng"),
            ["[eng:result] world", "[eng:result] second"],
        )

    def test_list_content(self):
        ev = self._user({"type": "tool_result", "content": [{"type": "text", "text": "ok line"}]})
        self.assertEqual(streaming.tool_result_lines(ev, "eng"), ["[eng:result] ok line"])

    def test_list_content_multiple_text_blocks_joined_by_newline(self):
        # tool_result.content can carry multiple text blocks; each one's
        # lines should be surfaced (joined with a real newline, not a space,
        # so multi-line file reads don't collapse into one terminal line).
        ev = self._user({"type": "tool_result", "content": [
            {"type": "text", "text": "1\tline-a"},
            {"type": "text", "text": "2\tline-b"},
        ]})
        self.assertEqual(
            streaming.tool_result_lines(ev, "eng"),
            ["[eng:result] 1\tline-a", "[eng:result] 2\tline-b"],
        )

    def test_error_tagged(self):
        ev = self._user({"type": "tool_result", "is_error": True, "content": "boom\nstack"})
        self.assertEqual(
            streaming.tool_result_lines(ev, "eng"),
            ["[eng:result-err] boom", "[eng:result-err] stack"],
        )


class PrinterTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("MLPAB_QUIET", None)

    def _capture(self, raw_line):
        printer = streaming.make_printer("agent")
        buf = io.StringIO()
        with redirect_stdout(buf):
            printer(raw_line)
        return buf.getvalue()

    def test_prints_assistant_tool_use(self):
        line = json.dumps(_assistant({"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}))
        self.assertEqual(self._capture(line), "[agent:bash] ls\n")

    def test_result_line(self):
        line = json.dumps({"type": "result", "num_turns": 7, "total_cost_usd": 0.1234, "subtype": "success"})
        out = self._capture(line)
        self.assertIn("[agent] done: 7 turns", out)
        self.assertIn("$0.1234", out)
        self.assertIn("stop=success", out)

    def test_malformed_json_is_ignored(self):
        self.assertEqual(self._capture("{not json"), "")


class TeeToTests(unittest.TestCase):

    def test_captures_python_and_subprocess_output(self):
        import subprocess
        import sys as _sys

        log = Path(tempfile.mkdtemp()) / "stream.log"
        # passthrough=False so the test's own terminal stays clean.
        with streaming.tee_to(log, passthrough=False):
            print("py-stdout-line", flush=True)
            print("py-stderr-line", file=_sys.stderr, flush=True)
            subprocess.run(["echo", "subprocess-line"], check=True)
        text = log.read_text()
        self.assertIn("py-stdout-line", text)
        self.assertIn("py-stderr-line", text)
        self.assertIn("subprocess-line", text)

    def test_restores_fds_after_block(self):
        import os as _os

        log = Path(tempfile.mkdtemp()) / "stream.log"
        before_out = _os.dup(1)
        before_err = _os.dup(2)
        try:
            with streaming.tee_to(log, passthrough=False):
                print("inside", flush=True)
            # After the block, fd 1/2 should point back at the originals: a fresh
            # dup should still be usable (no exception) and writes shouldn't land
            # in the log anymore.
            print("outside", flush=True)
            self.assertNotIn("outside", log.read_text())
        finally:
            _os.close(before_out)
            _os.close(before_err)


if __name__ == "__main__":
    unittest.main()
