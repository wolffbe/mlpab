"""Tests for the Codex agent engine: model routing, raw-event normalization
into the Claude-style transcript (so the existing accounting pipeline works
unchanged), and the per-run config.toml writer."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from mlpab import codex_runner, results


class AgentEnvTests(unittest.TestCase):
    """The agent's bash (run by codex) must find the interface binary, so the
    per-run venv bin has to be on PATH — otherwise CLI-mode `aws ...` calls die
    with `command not found`. Regression guard for that wiring."""

    def _run_capture_env(self, run_dir: Path):
        captured = {}

        class _FakeProc:
            stdout = iter(())

            def wait(self):
                return 0

            def kill(self):
                pass

        def _fake_popen(cmd, cwd, env, **kwargs):
            captured["env"] = env
            return _FakeProc()

        with mock.patch.object(codex_runner.shutil, "which", return_value="/usr/bin/codex"), \
             mock.patch.object(codex_runner.subprocess, "Popen", _fake_popen), \
             mock.patch.dict(os.environ, {"OPENAI_API_KEY": "x"}, clear=False):
            codex_runner.run(
                prompt="hi",
                run_dir=run_dir,
                auth="login",
                model="gpt-5.1-codex",
                cli_binary="aws",
                sdk_module=None,
                mcp_servers={},
                command_log=run_dir / "commands.jsonl",
                timeout_s=None,
            )
        return captured["env"]

    def test_per_run_venv_bin_on_path(self):
        run_dir = Path(tempfile.mkdtemp())
        venv_bin = run_dir / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        env = self._run_capture_env(run_dir)
        self.assertTrue(
            env["PATH"].startswith(f"{venv_bin}{os.pathsep}"),
            f"per-run venv bin not first on PATH: {env['PATH']!r}",
        )
        self.assertEqual(env["VIRTUAL_ENV"], str(run_dir / "venv"))

    def test_no_venv_dir_does_not_prepend(self):
        run_dir = Path(tempfile.mkdtemp())  # no venv/bin
        env = self._run_capture_env(run_dir)
        self.assertFalse(env["PATH"].startswith(f"{run_dir / 'venv' / 'bin'}{os.pathsep}"))


class ModelRoutingTests(unittest.TestCase):
    def test_codex_models_route_to_codex(self):
        for m in ("gpt-5.1-codex", "gpt-5-codex-mini", "codex-mini-latest", "gpt-4.1"):
            self.assertTrue(codex_runner.is_codex_model(m), m)

    def test_claude_models_route_to_claude(self):
        for m in ("claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5-20251001"):
            self.assertFalse(codex_runner.is_codex_model(m), m)

    def test_empty_model_routes_to_claude(self):
        self.assertFalse(codex_runner.is_codex_model(""))


def _write_events(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


class NormalizeTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.raw = self.dir / "codex_events.jsonl"
        self.transcript = self.dir / "transcript.jsonl"

    def _events(self):
        return [
            {"type": "thread.started", "thread_id": "t-1"},
            {"type": "turn.started"},
            {
                "type": "item.completed",
                "item": {"item_type": "command_execution", "command": "hops fg create --name x"},
            },
            {
                "type": "item.completed",
                "item": {"item_type": "command_execution", "command": "python train.py"},
            },
            {
                "type": "item.completed",
                "item": {"item_type": "mcp_tool_call", "server": "hopsworks", "tool": "create_fg"},
            },
            {"type": "item.completed", "item": {"item_type": "file_change"}},
            {"type": "item.completed", "item": {"item_type": "agent_message", "text": "done"}},
            {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 40}},
            {"type": "turn.completed", "usage": {"input_tokens": 250, "output_tokens": 60}},
        ]

    def test_usage_totals_and_turns(self):
        _write_events(self.raw, self._events())
        codex_runner.normalize_events(self.raw, self.transcript)
        usage = results.parse_transcript_usage(self.transcript)
        self.assertEqual(usage["input_tokens"], 350)
        self.assertEqual(usage["output_tokens"], 100)
        self.assertEqual(usage["total_tokens"], 450)
        self.assertEqual(usage["llm_calls"], 2)

    def test_command_accounting_via_existing_pipeline(self):
        _write_events(self.raw, self._events())
        codex_runner.normalize_events(self.raw, self.transcript)
        counts = results.aggregate_commands(
            self.transcript,
            cli_binary="hops",
            run_dir=self.dir,
        )
        self.assertEqual(counts["cli_calls"], 1)  # hops fg create
        self.assertEqual(counts["python_calls"], 1)  # python train.py
        self.assertEqual(counts["mcp_calls"], 1)  # mcp__hopsworks__create_fg
        self.assertEqual(counts["edit_calls"], 1)  # file_change → Edit

    def test_agent_message_not_a_tool_call(self):
        _write_events(self.raw, self._events())
        codex_runner.normalize_events(self.raw, self.transcript)
        lines = [json.loads(l) for l in self.transcript.read_text().splitlines()]
        tool_events = [l for l in lines if l.get("type") == "assistant"]
        self.assertEqual(len(tool_events), 4)  # 2 bash + 1 mcp + 1 file_change

    def test_missing_raw_still_writes_result_event(self):
        codex_runner.normalize_events(self.dir / "absent.jsonl", self.transcript)
        usage = results.parse_transcript_usage(self.transcript)
        self.assertEqual(usage["total_tokens"], 0)
        self.assertEqual(usage["llm_calls"], 0)

    def test_failed_command_exit_codes_counted(self):
        # Non-zero exit_code on a command_execution → an errored tool_result in
        # the normalized transcript → failed_commands. Codex has no hook, so
        # denied_calls stays 0.
        _write_events(
            self.raw,
            [
                {
                    "type": "item.completed",
                    "item": {
                        "item_type": "command_execution",
                        "command": "hops bad",
                        "exit_code": 1,
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "item_type": "command_execution",
                        "command": "hops ok",
                        "exit_code": 0,
                    },
                },
                {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
            ],
        )
        codex_runner.normalize_events(self.raw, self.transcript)
        counts = results.aggregate_commands(self.transcript, cli_binary="hops", run_dir=self.dir)
        self.assertEqual(counts["failed_commands"], 1)
        self.assertEqual(counts["denied_calls"], 0)
        self.assertEqual(counts["cli_calls"], 2)  # attempts still counted

    def test_subcommand_entrypoint_accounting(self):
        # `aws sagemaker …` counts as cli; `aws s3 …` does not.
        _write_events(
            self.raw,
            [
                {
                    "type": "item.completed",
                    "item": {
                        "item_type": "command_execution",
                        "command": "aws sagemaker list-models",
                    },
                },
                {
                    "type": "item.completed",
                    "item": {"item_type": "command_execution", "command": "aws s3 ls"},
                },
                {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
            ],
        )
        codex_runner.normalize_events(self.raw, self.transcript)
        counts = results.aggregate_commands(
            self.transcript,
            cli_binary="aws",
            cli_subcommand="sagemaker",
            run_dir=self.dir,
        )
        self.assertEqual(counts["cli_calls"], 1)
        self.assertEqual(counts["bash_calls"], 1)


class ConfigTomlTests(unittest.TestCase):
    def test_config_has_sandbox_network_and_mcp(self):
        d = Path(tempfile.mkdtemp())
        cfg = codex_runner._write_codex_config(
            d / ".codex",
            {
                "hopsworks": {
                    "command": "hopsworks-mcp",
                    "args": ["--transport", "stdio"],
                    "env": {},
                }
            },
            run_dir=d,
        )
        text = cfg.read_text()
        self.assertIn('sandbox_mode = "workspace-write"', text)
        self.assertIn("network_access = true", text)
        self.assertIn("[mcp_servers.hopsworks]", text)
        self.assertIn('command = "hopsworks-mcp"', text)
        self.assertIn('args = ["--transport", "stdio"]', text)
        # Slow-starting servers (databricks UC) die under codex's 10s default.
        self.assertIn("startup_timeout_sec = 180", text)
        self.assertIn("tool_timeout_sec = 3600", text)

    def test_no_mcp_servers_is_fine(self):
        d = Path(tempfile.mkdtemp())
        cfg = codex_runner._write_codex_config(d / ".codex", {}, run_dir=d)
        self.assertNotIn("[mcp_servers", cfg.read_text())


class RateLimitDetectionTests(unittest.TestCase):
    """`_rate_limited` decides whether a failed codex run gets retried with
    back-off. It must fire on transient/rate-limit conditions and NOT on ordinary
    failures (which would burn the 5h15m retry window for nothing)."""

    def _files(self, raw_lines, stderr=""):
        d = Path(tempfile.mkdtemp())
        raw = d / "codex_events.jsonl"
        raw.write_text("\n".join(raw_lines) + ("\n" if raw_lines else ""))
        err = d / "codex.stderr.log"
        err.write_text(stderr)
        return raw, err

    def test_error_event_with_429_is_rate_limited(self):
        raw, err = self._files([json.dumps({"type": "error", "message": "HTTP 429 Too Many Requests"})])
        self.assertTrue(codex_runner._rate_limited(raw, err))

    def test_error_event_overloaded_is_rate_limited(self):
        raw, err = self._files([json.dumps({"type": "error", "message": "server overloaded, retry"})])
        self.assertTrue(codex_runner._rate_limited(raw, err))

    def test_stderr_rate_limit_is_detected(self):
        raw, err = self._files([], stderr="error: rate limit exceeded for org")
        self.assertTrue(codex_runner._rate_limited(raw, err))

    def test_ordinary_error_is_not_rate_limited(self):
        raw, err = self._files([json.dumps({"type": "error", "message": "model_not_found: bad id"})])
        self.assertFalse(codex_runner._rate_limited(raw, err))

    def test_bare_number_in_normal_output_is_not_rate_limited(self):
        # A 500 appearing as a line number / token count must not look transient.
        raw, err = self._files(
            [json.dumps({"type": "item.completed", "item": {"command": "head -n 500 f"}})]
        )
        self.assertFalse(codex_runner._rate_limited(raw, err))


class PrintEventTests(unittest.TestCase):
    """`_print_event` renders codex `exec --json` events as Claude-style
    `[agent:…]` lines (parity with claude_runner's live pane / agent.log)."""

    def setUp(self):
        os.environ.pop("MLPAB_QUIET", None)

    def _capture(self, event):
        buf = io.StringIO()
        with redirect_stdout(buf):
            codex_runner._print_event(json.dumps(event))
        return buf.getvalue()

    def test_command_renders_bash_then_result(self):
        out = self._capture(
            {
                "type": "item.completed",
                "item": {
                    "item_type": "command_execution",
                    "command": "aws s3 ls",
                    "aggregated_output": "bucket-a\nbucket-b",
                    "exit_code": 0,
                },
            }
        )
        self.assertEqual(
            out,
            "[agent:bash] aws s3 ls\n[agent:result] bucket-a\n[agent:result] bucket-b\n",
        )

    def test_failed_command_tags_exit_code(self):
        out = self._capture(
            {
                "type": "item.completed",
                "item": {"item_type": "command_execution", "command": "aws boom", "exit_code": 2},
            }
        )
        self.assertEqual(out, "[agent:bash] aws boom\n[agent:result-err] exit 2\n")

    def test_agent_message_rendered(self):
        out = self._capture(
            {"type": "item.completed", "item": {"item_type": "agent_message", "text": "hi\nthere"}}
        )
        self.assertEqual(out, "[agent] hi\n[agent] there\n")

    def test_mcp_tool_rendered(self):
        out = self._capture(
            {
                "type": "item.completed",
                "item": {"item_type": "mcp_tool_call", "server": "hops", "tool": "list"},
            }
        )
        self.assertEqual(out, "[agent:mcp] mcp__hops__list\n")

    def test_error_event_tagged(self):
        out = self._capture({"type": "error", "message": "429 slow down"})
        self.assertEqual(out, "[agent:result-err] 429 slow down\n")

    def test_is_failed_exit_coerces_string_and_int(self):
        f = codex_runner._is_failed_exit
        self.assertFalse(f(None))
        self.assertFalse(f(0))
        self.assertFalse(f("0"))   # codex may emit exit_code as a string
        self.assertTrue(f(1))
        self.assertTrue(f("2"))
        self.assertFalse(f(""))    # missing/empty → not a failure

    def test_turn_completed_is_silent(self):
        # claude prints only on the final `result` event, not per turn.
        out = self._capture({"type": "turn.completed", "usage": {"input_tokens": 5}})
        self.assertEqual(out, "")

    def test_malformed_json_ignored(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            codex_runner._print_event("{not json")
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
