"""Unit tests for the Mistral Vibe agent engine: model routing (mutually
exclusive with codex/claude) and the streaming-event → transcript normalizer."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mlpab import codex_runner, mistral_runner


class ModelRoutingTests(unittest.TestCase):
    def test_recognises_mistral(self):
        for m in ("mistral-medium-3.5", "mistral-small-4", "mistral-large-3"):
            self.assertTrue(mistral_runner.is_mistral_model(m), m)

    def test_rejects_other_engines(self):
        for m in ("claude-opus-4-8", "gpt-5.5", "gpt-5.1-codex", "fable-5"):
            self.assertFalse(mistral_runner.is_mistral_model(m), m)

    def test_mutually_exclusive_with_codex(self):
        # the dispatch in runner.py checks codex FIRST then mistral — ensure no id
        # is claimed by both engines.
        for m in ("mistral-medium-3.5", "mistral-large-3", "gpt-5.5", "claude-opus-4-8"):
            self.assertFalse(
                mistral_runner.is_mistral_model(m) and codex_runner.is_codex_model(m), m
            )


class AgentEnvTests(unittest.TestCase):
    """The agent's bash (run by vibe) must find the interface binary, so the
    per-run venv bin has to be on PATH — otherwise CLI-mode `aws ...` calls die
    with `command not found`. Regression guard for that wiring."""

    def _run_capture_env(self, run_dir: Path):
        """Invoke mistral_runner.run with vibe/subprocess mocked; return the env
        passed to the spawned process."""
        captured = {}

        class _FakeProc:
            stdout = iter(())  # no streamed events

            def wait(self):
                return 0

            def kill(self):
                pass

        def _fake_popen(cmd, cwd, env, **kwargs):
            captured["env"] = env
            return _FakeProc()

        with mock.patch.object(mistral_runner.shutil, "which", return_value="/usr/bin/vibe"), \
             mock.patch.object(mistral_runner.subprocess, "Popen", _fake_popen), \
             mock.patch.dict(os.environ, {"MISTRAL_API_KEY": "x"}, clear=False):
            mistral_runner.run(
                prompt="hi",
                run_dir=run_dir,
                auth="login",
                model="mistral-large-3",
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


class NormalizeEventsTests(unittest.TestCase):
    def _normalize(self, events):
        d = Path(tempfile.mkdtemp())
        raw = d / "vibe_events.jsonl"
        raw.write_text("\n".join(json.dumps(e) for e in events))
        out = d / "transcript.jsonl"
        mistral_runner.normalize_events(raw, out)
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        result = next(l for l in lines if l["type"] == "result")
        return lines, result

    def test_vibe_messages_map_to_tool_use_and_turns(self):
        # vibe --output streaming emits role-based messages (assistant w/ tool_calls,
        # tool results); no usage in the stream.
        lines, result = self._normalize(
            [
                {
                    "role": "assistant",
                    "content": "running",
                    "tool_calls": [
                        {"function": {"name": "bash", "arguments": '{"command": "ls -la"}'}}
                    ],
                },
                {"role": "tool", "content": "file1\nfile2"},
                {"role": "assistant", "content": "done"},
            ]
        )
        tools = [l for l in lines if l["type"] == "assistant"]
        block = tools[0]["message"]["content"][0]
        self.assertEqual(block["name"], "Bash")
        self.assertEqual(block["input"]["command"], "ls -la")
        self.assertEqual(result["num_turns"], 2)  # two assistant messages
        self.assertEqual(result["usage"], {"input_tokens": 0, "output_tokens": 0})  # no meta

    def test_explicit_error_signal_flagged(self):
        # vibe's own error flag (or status) marks the tool result as an error.
        for msg in (
            {"role": "tool", "content": "boom", "is_error": True},
            {"role": "tool", "content": "boom", "status": "error"},
        ):
            lines, _ = self._normalize([msg])
            errs = [l for l in lines if l["type"] == "user"]
            self.assertTrue(errs and errs[0]["message"]["content"][0]["is_error"])

    def test_success_content_starting_with_error_not_flagged(self):
        # A successful result whose text merely starts with "error" (reading a
        # file about error handling, an echoed log line) must NOT be counted as
        # a failed command — only an explicit error signal does.
        lines, _ = self._normalize([{"role": "tool", "content": "Error: command failed"}])
        errs = [l for l in lines if l["type"] == "user"]
        self.assertTrue(errs)
        self.assertFalse(errs[0]["message"]["content"][0]["is_error"])

    def test_tokens_from_session_meta(self):
        import json as _json

        d = Path(tempfile.mkdtemp())
        (d / "vibe_events.jsonl").write_text("")
        meta = d / ".vibe" / "logs" / "session" / "s1"
        meta.mkdir(parents=True)
        (meta / "meta.json").write_text(
            _json.dumps({"session_prompt_tokens": 1234, "session_completion_tokens": 56})
        )
        out = d / "transcript.jsonl"
        mistral_runner.normalize_events(d / "vibe_events.jsonl", out)
        result = [json.loads(l) for l in out.read_text().splitlines()][-1]
        self.assertEqual(result["usage"], {"input_tokens": 1234, "output_tokens": 56})

    def test_empty_events_still_yield_result(self):
        _, result = self._normalize([])
        self.assertEqual(result["usage"], {"input_tokens": 0, "output_tokens": 0})


class RateLimitDetectionTests(unittest.TestCase):
    """`_rate_limited` decides whether a failed vibe run gets retried with
    back-off — fire on transient/rate-limit conditions, not ordinary failures."""

    def _files(self, raw_lines, stderr=""):
        d = Path(tempfile.mkdtemp())
        raw = d / "vibe_events.jsonl"
        raw.write_text("\n".join(raw_lines) + ("\n" if raw_lines else ""))
        err = d / "vibe.stderr.log"
        err.write_text(stderr)
        return raw, err

    def test_type_error_event_is_rate_limited(self):
        raw, err = self._files([json.dumps({"type": "error", "message": "429 too many requests"})])
        self.assertTrue(mistral_runner._rate_limited(raw, err))

    def test_role_error_content_is_rate_limited(self):
        raw, err = self._files([json.dumps({"role": "error", "content": "rate_limit hit, slow down"})])
        self.assertTrue(mistral_runner._rate_limited(raw, err))

    def test_stderr_rate_limit_is_detected(self):
        raw, err = self._files([], stderr="mistral: service unavailable (529)")
        self.assertTrue(mistral_runner._rate_limited(raw, err))

    def test_ordinary_assistant_message_is_not_rate_limited(self):
        raw, err = self._files([json.dumps({"role": "assistant", "content": "the answer is 429"})])
        self.assertFalse(mistral_runner._rate_limited(raw, err))

    def test_ordinary_error_is_not_rate_limited(self):
        raw, err = self._files([json.dumps({"type": "error", "message": "invalid api key"})])
        self.assertFalse(mistral_runner._rate_limited(raw, err))


if __name__ == "__main__":
    unittest.main()
