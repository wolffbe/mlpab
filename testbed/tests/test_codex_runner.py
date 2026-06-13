"""Tests for the Codex agent engine: model routing, raw-event normalization
into the Claude-style transcript (so the existing accounting pipeline works
unchanged), and the per-run config.toml writer."""
import json
import tempfile
import unittest
from pathlib import Path

from mlpab import codex_runner, results


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
            {"type": "item.completed",
             "item": {"item_type": "command_execution", "command": "hops fg create --name x"}},
            {"type": "item.completed",
             "item": {"item_type": "command_execution", "command": "python train.py"}},
            {"type": "item.completed",
             "item": {"item_type": "mcp_tool_call", "server": "hopsworks", "tool": "create_fg"}},
            {"type": "item.completed",
             "item": {"item_type": "file_change"}},
            {"type": "item.completed",
             "item": {"item_type": "agent_message", "text": "done"}},
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
            self.transcript, cli_binary="hops", run_dir=self.dir,
        )
        self.assertEqual(counts["cli_calls"], 1)      # hops fg create
        self.assertEqual(counts["python_calls"], 1)   # python train.py
        self.assertEqual(counts["mcp_calls"], 1)      # mcp__hopsworks__create_fg
        self.assertEqual(counts["edit_calls"], 1)        # file_change → Edit

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
        _write_events(self.raw, [
            {"type": "item.completed",
             "item": {"item_type": "command_execution", "command": "hops bad", "exit_code": 1}},
            {"type": "item.completed",
             "item": {"item_type": "command_execution", "command": "hops ok", "exit_code": 0}},
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ])
        codex_runner.normalize_events(self.raw, self.transcript)
        counts = results.aggregate_commands(self.transcript, cli_binary="hops", run_dir=self.dir)
        self.assertEqual(counts["failed_commands"], 1)
        self.assertEqual(counts["denied_calls"], 0)
        self.assertEqual(counts["cli_calls"], 2)   # attempts still counted

    def test_subcommand_entrypoint_accounting(self):
        # `aws sagemaker …` counts as cli; `aws s3 …` does not.
        _write_events(self.raw, [
            {"type": "item.completed",
             "item": {"item_type": "command_execution", "command": "aws sagemaker list-models"}},
            {"type": "item.completed",
             "item": {"item_type": "command_execution", "command": "aws s3 ls"}},
            {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
        ])
        codex_runner.normalize_events(self.raw, self.transcript)
        counts = results.aggregate_commands(
            self.transcript, cli_binary="aws", cli_subcommand="sagemaker", run_dir=self.dir,
        )
        self.assertEqual(counts["cli_calls"], 1)
        self.assertEqual(counts["bash_calls"], 1)


class ConfigTomlTests(unittest.TestCase):
    def test_config_has_sandbox_network_and_mcp(self):
        d = Path(tempfile.mkdtemp())
        cfg = codex_runner._write_codex_config(
            d / ".codex",
            {"hopsworks": {"command": "hopsworks-mcp",
                           "args": ["--transport", "stdio"], "env": {}}},
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


if __name__ == "__main__":
    unittest.main()
