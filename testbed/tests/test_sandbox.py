"""Unit tests for the confinement design.

Covers: settings.json shape (sandbox + denies), OAuth token Keychain helper,
agent env construction (HOME redirect + auth fallback). Live-claude
integration lives in tests/integration/live_sandbox.py.
"""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mlpab import claude_runner
from mlpab.hooks import log_tool_call as hook

# ---------------------------------------------------------------------------
# 1) Static settings shape
# ---------------------------------------------------------------------------


class DenyPatternsTests(unittest.TestCase):
    """Patterns that DO fire in bypassPermissions: Bash escapes only.

    Read/Write/Edit path denies are silently skipped in bypass mode — those
    are enforced by the PreToolUse hook (see hooks/log_tool_call.py + the
    `hook_*` tests below) instead.
    """

    def test_blocks_bash_relative_traversal(self):
        p = claude_runner.deny_patterns_for(Path("/some/run/v0"))
        self.assertIn("Bash(cd *..*)", p)

    def test_blocks_bash_cat_ls_into_home_dotfiles(self):
        p = claude_runner.deny_patterns_for(Path("/some/run"))
        home = str(claude_runner.HOME_DIR)
        self.assertIn(f"Bash(cat {home}/.*)", p)
        self.assertIn(f"Bash(cat {home}/.*/**)", p)
        self.assertIn(f"Bash(ls {home}/.*)", p)
        self.assertIn(f"Bash(ls {home}/.*/**)", p)

    def test_no_bypass_skipped_path_patterns(self):
        # Read/Write/Edit path patterns would not fire in bypassPermissions,
        # so we don't emit them — they were dead weight + misleading
        # documentation. The hook handles Read/Write/Edit boundary checks.
        p = claude_runner.deny_patterns_for(Path("/some/run"))
        for tool in ("Read", "Write", "Edit"):
            self.assertFalse(
                any(s.startswith(f"{tool}(") for s in p),
                f"{tool}(...) deny patterns should not be emitted",
            )


class WriteSettingsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name) / "run"
        self.run_dir.mkdir()
        self.command_log = self.run_dir / "commands.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def _load(self):
        return json.loads((self.run_dir / ".claude" / "settings.json").read_text())

    def test_sandbox_network_gated_filesystem_open(self):
        # Agent's sandbox: network is gated (allowlist of claude API +
        # loopback + per-interface domains), filesystem is explicitly opted
        # OUT of restriction (`allowRead: ["/"]` + `allowWrite: ["/"]`).
        # Enabling the sandbox at all engages default fs restrictions,
        # which broke legit Bash mkdir; opening fs back up while keeping
        # network gated is the only shape we found that works.
        claude_runner._write_settings(self.run_dir, self.command_log)
        sb = self._load()["sandbox"]
        self.assertTrue(sb["enabled"])
        # Filesystem fully open.
        self.assertEqual(sb["filesystem"]["allowRead"], ["/"])
        self.assertEqual(sb["filesystem"]["allowWrite"], ["/"])
        # Network gated to baseline.
        net = sb["network"]
        self.assertTrue(net["allowLocalBinding"])
        for h in ("127.0.0.1", "api.anthropic.com"):
            self.assertIn(h, net["allowedDomains"])
        for blocked in ("pypi.org", "huggingface.co", "example.com", "github.com"):
            self.assertNotIn(blocked, net["allowedDomains"])

    def test_sandbox_merges_interface_allowed_domains(self):
        # `allowed_domains=[...]` passed in (from interface config) is appended
        # to the baseline allowlist.
        claude_runner._write_settings(
            self.run_dir, self.command_log, allowed_domains=["api.openai.com", "*.openai.com"]
        )
        net = self._load()["sandbox"]["network"]
        self.assertIn("api.openai.com", net["allowedDomains"])
        self.assertIn("*.openai.com", net["allowedDomains"])
        # Baseline still present.
        self.assertIn("api.anthropic.com", net["allowedDomains"])

    def test_sandbox_excluded_commands_passed_through(self):
        # Go binaries (databricks CLI) can't verify TLS inside Seatbelt
        # (trustd unreachable → OSStatus -26276); the manifest's
        # `sandbox_excluded_commands` must land in sandbox.excludedCommands
        # so they run outside the sandbox.
        claude_runner._write_settings(
            self.run_dir, self.command_log, sandbox_excluded_commands=["databricks", "databricks *"]
        )
        sb = self._load()["sandbox"]
        self.assertEqual(sb["excludedCommands"], ["databricks", "databricks *"])

    def test_sandbox_excluded_commands_default_empty(self):
        # No manifest key → nothing runs outside the sandbox.
        claude_runner._write_settings(self.run_dir, self.command_log)
        self.assertEqual(self._load()["sandbox"]["excludedCommands"], [])

    def test_hook_script_copied_into_boundary(self):
        # The PreToolUse hook command should reference an in-boundary script,
        # not the testbed source path (would require an allowRead exception).
        claude_runner._write_settings(self.run_dir, self.command_log)
        hook_cmd = self._load()["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        self.assertTrue(
            hook_cmd.endswith("/.claude/hooks/log_tool_call.py"),
            f"hook command should point inside boundary; got {hook_cmd!r}",
        )
        self.assertNotIn(str(claude_runner.TESTBED_ROOT), hook_cmd)
        # And the copied script actually exists + is executable.
        local = self.run_dir / ".claude" / "hooks" / "log_tool_call.py"
        self.assertTrue(local.is_file())
        self.assertTrue(local.stat().st_mode & 0o111)

    def test_home_dir_is_real_user_home_not_redirected(self):
        # Regression: HOME_DIR must come from /etc/passwd (pwd module), NOT
        # $HOME — the runner redirects $HOME into the run dir before
        # claude_runner is imported in nested invocations. A redirected
        # HOME_DIR would emit denyRead rooted at <run>/, which then blocks
        # parent-dir lookups for writes inside the agent's own cwd.
        import pwd as _pwd

        real_home = Path(_pwd.getpwuid(os.getuid()).pw_dir).resolve()
        self.assertEqual(claude_runner.HOME_DIR, real_home)

    def test_hooks_pretooluse_command_present(self):
        claude_runner._write_settings(self.run_dir, self.command_log)
        hooks = self._load()["hooks"]["PreToolUse"]
        self.assertEqual(len(hooks), 1)
        self.assertEqual(hooks[0]["matcher"], ".*")
        self.assertIn("log_tool_call.py", hooks[0]["hooks"][0]["command"])

    def test_agent_denies_include_common_agent_and_escape_patterns(self):
        claude_runner._write_settings(self.run_dir, self.command_log)
        denies = self._load()["permissions"]["deny"]
        for t in claude_runner.COMMON_DENY:
            self.assertIn(t, denies)
        for t in claude_runner.AGENT_ONLY_DENY:
            self.assertIn(t, denies)
        for pat in claude_runner.deny_patterns_for(self.run_dir.resolve()):
            self.assertIn(pat, denies)

    def test_settings_written_to_run_dir(self):
        claude_runner._write_settings(self.run_dir, self.command_log)
        self.assertTrue((self.run_dir / ".claude" / "settings.json").exists())


# ---------------------------------------------------------------------------
# 2) OAuth / env construction
# ---------------------------------------------------------------------------


class OAuthTokenFromKeychainTests(unittest.TestCase):
    def test_parses_access_token_from_security_output(self):
        payload = json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "sk-ant-oat01-EXAMPLE-TOKEN",
                    "refreshToken": "sk-ant-ort01-EXAMPLE-REFRESH",
                }
            }
        )
        with mock.patch("subprocess.run", return_value=mock.Mock(stdout=payload)) as p:
            self.assertEqual(
                claude_runner.oauth_token_from_keychain(), "sk-ant-oat01-EXAMPLE-TOKEN"
            )
            args = p.call_args.args[0]
            self.assertEqual(args[:3], ["/usr/bin/security", "find-generic-password", "-s"])
            self.assertIn("Claude Code-credentials", args)

    def test_missing_entry_returns_none(self):
        with mock.patch(
            "subprocess.run", side_effect=subprocess.CalledProcessError(44, "security")
        ):
            self.assertIsNone(claude_runner.oauth_token_from_keychain())

    def test_security_binary_missing_returns_none(self):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError()):
            self.assertIsNone(claude_runner.oauth_token_from_keychain())

    def test_malformed_payload_returns_none(self):
        with mock.patch("subprocess.run", return_value=mock.Mock(stdout="not json\n")):
            self.assertIsNone(claude_runner.oauth_token_from_keychain())


class TokenCacheTests(unittest.TestCase):
    """Per-run token cache: side-channel around Claude Code's env stripping.

    The cache file lives at `<run>/.claude-oauth` (mode 0600); the env var
    `MLPAB_TOKEN_CACHE` points downstream mlpab invocations at it.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_write_and_read_round_trip(self):
        cache = claude_runner.write_token_cache("sk-ant-oat01-CACHED", self.run_path)
        # Lives at <run>/.claude-oauth, mode 0600.
        self.assertEqual(cache, self.run_path / claude_runner.TOKEN_CACHE_FILENAME)
        self.assertEqual(cache.stat().st_mode & 0o777, 0o600)
        # Readable via MLPAB_TOKEN_CACHE pointing at the file.
        with mock.patch.dict(os.environ, {claude_runner.TOKEN_CACHE_ENV: str(cache)}, clear=False):
            self.assertEqual(claude_runner.read_token_cache(), "sk-ant-oat01-CACHED")

    def test_read_missing_env_returns_none(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(claude_runner.read_token_cache())

    def test_read_missing_file_returns_none(self):
        with mock.patch.dict(
            os.environ, {claude_runner.TOKEN_CACHE_ENV: "/nonexistent/path"}, clear=True
        ):
            self.assertIsNone(claude_runner.read_token_cache())

    def test_resolve_order_env_beats_keychain_beats_cache(self):
        # env → keychain → cache; first hit wins. The Keychain is fresher than
        # the session-start cache snapshot (an interactive claude refreshes it),
        # so it must win over the cache — preferring the stale cache once froze
        # an expired token for ~30 runs.
        cache = claude_runner.write_token_cache("CACHED", self.run_path)
        with mock.patch.dict(
            os.environ,
            {"CLAUDE_CODE_OAUTH_TOKEN": "ENV", claude_runner.TOKEN_CACHE_ENV: str(cache)},
            clear=True,
        ):
            self.assertEqual(claude_runner.resolve_oauth_token(), "ENV")
        with (
            mock.patch.dict(os.environ, {claude_runner.TOKEN_CACHE_ENV: str(cache)}, clear=True),
            mock.patch.object(claude_runner, "oauth_token_from_keychain", return_value="KEYCHAIN"),
        ):
            self.assertEqual(claude_runner.resolve_oauth_token(), "KEYCHAIN")
        with (
            mock.patch.dict(os.environ, {claude_runner.TOKEN_CACHE_ENV: str(cache)}, clear=True),
            mock.patch.object(claude_runner, "oauth_token_from_keychain", return_value=None),
        ):
            self.assertEqual(claude_runner.resolve_oauth_token(), "CACHED")
        cache.unlink()
        with (
            mock.patch.dict(os.environ, {claude_runner.TOKEN_CACHE_ENV: str(cache)}, clear=True),
            mock.patch.object(claude_runner, "oauth_token_from_keychain", return_value="KEYCHAIN"),
        ):
            self.assertEqual(claude_runner.resolve_oauth_token(), "KEYCHAIN")


class AgentEnvConstructionTests(unittest.TestCase):
    """Patch run_with_retry to capture the env dict claude_runner.run builds."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name) / "run"
        self.run_dir.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _invoke(self, auth: str = "login", token: str | None = "sk-ant-oat01-FAKE"):
        captured = {}

        def fake_run_with_retry(*, cmd, cwd, env, **kw):
            captured.update(cmd=cmd, cwd=cwd, env=env)
            return 0, 0.01, 0.0

        # Force keychain + token cache to the mocked value so a real on-disk
        # cache file from a previous run doesn't leak into tests.
        with (
            mock.patch("shutil.which", return_value="/usr/bin/claude"),
            mock.patch.object(claude_runner, "run_with_retry", side_effect=fake_run_with_retry),
            mock.patch.object(claude_runner, "oauth_token_from_keychain", return_value=token),
            mock.patch.object(claude_runner, "read_token_cache", return_value=None),
        ):
            claude_runner.run(
                prompt="hi",
                run_dir=self.run_dir,
                auth=auth,
                model="claude-sonnet-4-6",
                cli_binary=None,
                sdk_module=None,
                mcp_servers={},
                command_log=self.run_dir / "commands.jsonl",
            )
        return captured

    def test_home_redirected_to_run_dir(self):
        self.assertEqual(self._invoke()["env"]["HOME"], str(self.run_dir.resolve()))

    def test_keychain_token_injected_when_auth_login(self):
        # Redirected HOME → claude cannot reach the Keychain itself; the
        # runner injects the freshest resolvable token (re-read every run).
        env = self._invoke(auth="login", token="sk-ant-oat01-TOKEN")["env"]
        self.assertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "sk-ant-oat01-TOKEN")
        self.assertNotIn("ANTHROPIC_API_KEY", env)  # login strips it

    def test_background_tasks_disabled_and_bash_timeouts_raised(self):
        # Auto-backgrounding a long command wedges `-p` mode (no completion
        # re-invoke). Disable it outright; raising BASH_*_TIMEOUT_MS alone does
        # NOT prevent backgrounding (only the kill).
        env = self._invoke()["env"]
        self.assertEqual(env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"], "1")
        self.assertIn("BASH_MAX_TIMEOUT_MS", env)
        self.assertIn("BASH_DEFAULT_TIMEOUT_MS", env)

    def test_mcp_timeouts_raised(self):
        # Slow-starting stdio MCP servers (databricks UC: ~20s of imports
        # before tool enumeration) die under Claude Code's default 30s startup
        # timeout → zero mcp__* tools registered. Startup gets a fixed generous
        # window; tool calls are pinned to the run budget like Bash.
        env = self._invoke()["env"]
        self.assertEqual(env["MCP_TIMEOUT"], str(180 * 1000))
        self.assertEqual(env["MCP_TOOL_TIMEOUT"], env["BASH_MAX_TIMEOUT_MS"])

    def test_warns_when_no_auth_available(self):
        # No API key, no Keychain token, no token cache → warn (don't raise).
        import io
        from contextlib import redirect_stdout

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(claude_runner, "read_token_cache", return_value=None),
        ):
            buf = io.StringIO()
            with redirect_stdout(buf):
                self._invoke(auth="login", token=None)
            self.assertIn("no auth available", buf.getvalue())

    def test_keychain_fallback_when_api_key_unset(self):
        # api-key mode without ANTHROPIC_API_KEY still auths via the
        # Keychain-resolved token.
        with mock.patch.dict(os.environ, {}, clear=True):
            env = self._invoke(auth="api-key", token="sk-ant-oat01-FALLBACK")["env"]
        self.assertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "sk-ant-oat01-FALLBACK")

    def test_inherited_env_token_used_when_no_keychain(self):
        # No Keychain credential → an inherited CLAUDE_CODE_OAUTH_TOKEN from
        # the parent env is re-injected via resolve_oauth_token().
        with mock.patch.dict(
            os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-INHERITED"}, clear=True
        ):
            captured = self._invoke(auth="login", token=None)
        self.assertEqual(captured["env"]["CLAUDE_CODE_OAUTH_TOKEN"], "sk-ant-oat01-INHERITED")

    def test_api_key_auth_keeps_api_key_and_injects_oauth_fallback(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-api01-USER"}, clear=False):
            env = self._invoke(auth="api-key", token="sk-ant-oat01-FALLBACK")["env"]
        self.assertEqual(env["ANTHROPIC_API_KEY"], "sk-ant-api01-USER")
        self.assertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "sk-ant-oat01-FALLBACK")

    def test_bash_timeouts_pinned_to_run_budget(self):
        # Long foreground commands must stay synchronous: both Bash timeouts
        # are pinned to the agent's whole `claude -p` budget (default
        # timeout_s = 3600s → 3_600_000 ms) so Claude Code never auto-moves a
        # training command to a background task mid-run.
        env = self._invoke()["env"]
        self.assertEqual(env["BASH_DEFAULT_TIMEOUT_MS"], str(60 * 60 * 1000))
        self.assertEqual(env["BASH_MAX_TIMEOUT_MS"], str(60 * 60 * 1000))


# ---------------------------------------------------------------------------
# 3) PreToolUse hook — boundary path checks
# ---------------------------------------------------------------------------


class HookTaskOutputAllowanceTests(unittest.TestCase):
    """The boundary hook denies reads outside the agent's cwd, EXCEPT for
    Claude Code's own background-task output files for THIS run — otherwise an
    auto-backgrounded command leaves the agent blind to its own output. The
    allowance is scoped by the cwd slug embedded in the task path, so a
    sibling run's task output stays denied.
    """

    def _check(self, path: str, boundary: str, cwd: str) -> str | None:
        with (
            mock.patch.dict(os.environ, {"TESTBED_BOUNDARY": boundary}, clear=False),
            mock.patch("os.getcwd", return_value=cwd),
        ):
            return hook._path_violates_boundary(path)

    def test_allows_own_background_task_output(self):
        cwd = "/runs/v0/image_classification/aerial-cactus"
        slug = cwd.replace("/", "-")
        path = f"/private/tmp/claude-501/{slug}/9a78-uuid/tasks/btph5a74.output"
        self.assertIsNone(self._check(path, boundary=cwd, cwd=cwd))

    def test_denies_sibling_run_task_output(self):
        cwd = "/runs/v0/image_classification/aerial-cactus"
        other = "/runs/v0/image_classification/some-other-task"
        other_slug = other.replace("/", "-")
        path = f"/private/tmp/claude-501/{other_slug}/uuid/tasks/x.output"
        self.assertIsNotNone(self._check(path, boundary=cwd, cwd=cwd))

    def test_denies_non_task_file_outside_boundary(self):
        cwd = "/runs/v0/image_classification/aerial-cactus"
        # Same temp tree, but not a /tasks/*.output file → still denied.
        slug = cwd.replace("/", "-")
        path = f"/private/tmp/claude-501/{slug}/uuid/some_other_file.txt"
        self.assertIsNotNone(self._check(path, boundary=cwd, cwd=cwd))

    def test_in_boundary_path_still_allowed(self):
        cwd = "/runs/v0/image_classification/aerial-cactus"
        self.assertIsNone(self._check(f"{cwd}/submission/submission.csv", boundary=cwd, cwd=cwd))


if __name__ == "__main__":
    unittest.main()
