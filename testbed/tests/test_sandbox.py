"""Unit tests for the confinement design.

Covers: settings.json shape (sandbox + denies), OAuth token Keychain helper,
engineer env construction (HOME redirect + auth fallback). Live-claude
integration lives in tests/integration/live_sandbox.py.
"""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from banter import claude_runner
from banter.hooks import log_tool_call as hook


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
        # Engineer's sandbox: network is gated (allowlist of claude API +
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
        for blocked in ("pypi.org", "huggingface.co", "kaggle.com", "github.com"):
            self.assertNotIn(blocked, net["allowedDomains"])

    def test_sandbox_merges_interface_allowed_domains(self):
        # `allowed_domains=[...]` passed in (from interface config) is appended
        # to the baseline allowlist.
        claude_runner._write_settings(self.run_dir, self.command_log,
                                      allowed_domains=["api.openai.com", "*.openai.com"])
        net = self._load()["sandbox"]["network"]
        self.assertIn("api.openai.com", net["allowedDomains"])
        self.assertIn("*.openai.com", net["allowedDomains"])
        # Baseline still present.
        self.assertIn("api.anthropic.com", net["allowedDomains"])

    def test_hook_script_copied_into_boundary(self):
        # The PreToolUse hook command should reference an in-boundary script,
        # not the testbed source path (would require an allowRead exception).
        claude_runner._write_settings(self.run_dir, self.command_log)
        hook_cmd = self._load()["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
        self.assertTrue(hook_cmd.endswith("/.claude/hooks/log_tool_call.py"),
                        f"hook command should point inside boundary; got {hook_cmd!r}")
        self.assertNotIn(str(claude_runner.TESTBED_ROOT), hook_cmd)
        # And the copied script actually exists + is executable.
        local = self.run_dir / ".claude" / "hooks" / "log_tool_call.py"
        self.assertTrue(local.is_file())
        self.assertTrue(local.stat().st_mode & 0o111)

    def test_home_dir_is_real_user_home_not_redirected(self):
        # Regression: HOME_DIR must come from /etc/passwd (pwd module), NOT
        # $HOME — because autoresearch redirects $HOME into the run dir
        # before banter run imports claude_runner. A redirected HOME_DIR
        # would emit denyRead rooted at <run>/, which then blocks parent-
        # dir lookups for writes inside the engineer's own cwd.
        import pwd as _pwd
        real_home = Path(_pwd.getpwuid(os.getuid()).pw_dir).resolve()
        self.assertEqual(claude_runner.HOME_DIR, real_home)

    def test_version_dir_accepted_for_backcompat(self):
        # `version_dir` is accepted as a kwarg but has no effect on the
        # emitted settings — the per-challenge engineer's boundary is just
        # its cwd; version_dir was a stale autoresearch concept.
        version_dir = self.run_dir / "v0"
        version_dir.mkdir()
        claude_runner._write_settings(self.run_dir, self.command_log, version_dir=version_dir)
        # Settings still write to <run>/.claude/, not the version dir.
        self.assertTrue((self.run_dir / ".claude" / "settings.json").exists())
        self.assertFalse((version_dir / ".claude" / "settings.json").exists())

    def test_hooks_pretooluse_command_present(self):
        claude_runner._write_settings(self.run_dir, self.command_log)
        hooks = self._load()["hooks"]["PreToolUse"]
        self.assertEqual(len(hooks), 1)
        self.assertEqual(hooks[0]["matcher"], ".*")
        self.assertIn("log_tool_call.py", hooks[0]["hooks"][0]["command"])

    def test_engineer_denies_include_common_engineer_and_escape_patterns(self):
        claude_runner._write_settings(self.run_dir, self.command_log)
        denies = self._load()["permissions"]["deny"]
        for t in claude_runner.COMMON_DENY:
            self.assertIn(t, denies)
        for t in claude_runner.ENGINEER_ONLY_DENY:
            self.assertIn(t, denies)
        for pat in claude_runner.deny_patterns_for(self.run_dir.resolve()):
            self.assertIn(pat, denies)

    def test_settings_written_to_run_dir_not_version_dir(self):
        version_dir = self.run_dir / "v0"
        version_dir.mkdir()
        claude_runner._write_settings(self.run_dir, self.command_log, version_dir=version_dir)
        self.assertTrue((self.run_dir / ".claude" / "settings.json").exists())
        self.assertFalse((version_dir / ".claude" / "settings.json").exists())


# ---------------------------------------------------------------------------
# 2) OAuth / env construction
# ---------------------------------------------------------------------------

class OAuthTokenFromKeychainTests(unittest.TestCase):
    def test_parses_access_token_from_security_output(self):
        payload = json.dumps({"claudeAiOauth": {
            "accessToken": "sk-ant-oat01-EXAMPLE-TOKEN",
            "refreshToken": "sk-ant-ort01-EXAMPLE-REFRESH",
        }})
        with mock.patch("subprocess.run", return_value=mock.Mock(stdout=payload)) as p:
            self.assertEqual(claude_runner.oauth_token_from_keychain(),
                             "sk-ant-oat01-EXAMPLE-TOKEN")
            args = p.call_args.args[0]
            self.assertEqual(args[:3], ["/usr/bin/security", "find-generic-password", "-s"])
            self.assertIn("Claude Code-credentials", args)

    def test_missing_entry_returns_none(self):
        with mock.patch("subprocess.run",
                        side_effect=subprocess.CalledProcessError(44, "security")):
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
    `BANTER_TOKEN_CACHE` points downstream banter invocations at it.
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
        # Readable via BANTER_TOKEN_CACHE pointing at the file.
        with mock.patch.dict(os.environ,
                             {claude_runner.TOKEN_CACHE_ENV: str(cache)},
                             clear=False):
            self.assertEqual(claude_runner.read_token_cache(), "sk-ant-oat01-CACHED")

    def test_read_missing_env_returns_none(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(claude_runner.read_token_cache())

    def test_read_missing_file_returns_none(self):
        with mock.patch.dict(os.environ,
                             {claude_runner.TOKEN_CACHE_ENV: "/nonexistent/path"},
                             clear=True):
            self.assertIsNone(claude_runner.read_token_cache())

    def test_resolve_order_env_beats_cache_beats_keychain(self):
        # env → cache → keychain; first hit wins.
        cache = claude_runner.write_token_cache("CACHED", self.run_path)
        with mock.patch.dict(os.environ,
                             {"CLAUDE_CODE_OAUTH_TOKEN": "ENV",
                              claude_runner.TOKEN_CACHE_ENV: str(cache)},
                             clear=True):
            self.assertEqual(claude_runner.resolve_oauth_token(), "ENV")
        with mock.patch.dict(os.environ,
                             {claude_runner.TOKEN_CACHE_ENV: str(cache)},
                             clear=True), \
             mock.patch.object(claude_runner, "oauth_token_from_keychain",
                               return_value="KEYCHAIN"):
            self.assertEqual(claude_runner.resolve_oauth_token(), "CACHED")
        cache.unlink()
        with mock.patch.dict(os.environ,
                             {claude_runner.TOKEN_CACHE_ENV: str(cache)},
                             clear=True), \
             mock.patch.object(claude_runner, "oauth_token_from_keychain",
                               return_value="KEYCHAIN"):
            self.assertEqual(claude_runner.resolve_oauth_token(), "KEYCHAIN")


class EngineerEnvConstructionTests(unittest.TestCase):
    """Patch run_with_retry to capture the env dict claude_runner.run builds."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self._tmp.name) / "run"
        self.run_dir.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _invoke(self, auth: str = "login", version_dir: Path | None = None,
                token: str | None = "sk-ant-oat01-FAKE"):
        captured = {}

        def fake_run_with_retry(*, cmd, cwd, env, **kw):
            captured.update(cmd=cmd, cwd=cwd, env=env)
            return 0, 0.01

        # Force keychain + token cache to the mocked value so a real on-disk
        # cache file from `banter autoresearch` doesn't leak into tests.
        with mock.patch("shutil.which", return_value="/usr/bin/claude"), \
             mock.patch.object(claude_runner, "run_with_retry", side_effect=fake_run_with_retry), \
             mock.patch.object(claude_runner, "oauth_token_from_keychain", return_value=token), \
             mock.patch.object(claude_runner, "read_token_cache", return_value=None):
            claude_runner.run(
                prompt="hi", run_dir=self.run_dir,
                auth=auth, model="claude-sonnet-4-6",
                cli_binary=None, sdk_module=None, mcp_servers={},
                command_log=self.run_dir / "commands.jsonl",
                version_dir=version_dir,
            )
        return captured

    def test_home_redirected_to_run_dir_for_benchmark(self):
        self.assertEqual(self._invoke(version_dir=None)["env"]["HOME"],
                         str(self.run_dir.resolve()))

    def test_home_redirected_to_version_dir_for_autoresearch(self):
        version_dir = self.run_dir / "v3"
        version_dir.mkdir()
        self.assertEqual(self._invoke(version_dir=version_dir)["env"]["HOME"],
                         str(version_dir.resolve()))

    def test_oauth_token_injected_when_auth_login(self):
        env = self._invoke(auth="login", token="sk-ant-oat01-TOKEN")["env"]
        self.assertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "sk-ant-oat01-TOKEN")
        self.assertNotIn("ANTHROPIC_API_KEY", env)  # login strips it

    def test_warns_when_no_auth_available(self):
        # No API key, no Keychain token, no token cache → warn (don't raise).
        import io
        from contextlib import redirect_stdout
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(claude_runner, "read_token_cache", return_value=None):
            buf = io.StringIO()
            with redirect_stdout(buf):
                self._invoke(auth="login", token=None)
            self.assertIn("no auth available", buf.getvalue())

    def test_keychain_fallback_when_api_key_unset(self):
        # api-key mode without ANTHROPIC_API_KEY must still auth via Keychain.
        with mock.patch.dict(os.environ, {}, clear=True):
            env = self._invoke(auth="api-key", token="sk-ant-oat01-FALLBACK")["env"]
        self.assertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "sk-ant-oat01-FALLBACK")

    def test_inherited_env_token_preferred_over_keychain(self):
        # If parent env already has CLAUDE_CODE_OAUTH_TOKEN, don't re-fetch
        # from Keychain (avoids nested-subprocess ACL failures masking a
        # working inherited token).
        with mock.patch.dict(os.environ,
                             {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-INHERITED"},
                             clear=True):
            with mock.patch.object(claude_runner, "oauth_token_from_keychain",
                                   return_value="sk-ant-oat01-FRESH") as ko:
                # token kwarg is ignored — patch.object above wins for the
                # actual keychain helper call. We just need to ensure run()
                # doesn't crash.
                captured = self._invoke(auth="login", token="ignored")
                # Keychain not consulted because env had the token already.
                ko.assert_not_called()
        self.assertEqual(captured["env"]["CLAUDE_CODE_OAUTH_TOKEN"], "sk-ant-oat01-INHERITED")

    def test_api_key_auth_keeps_api_key_and_still_injects_oauth_fallback(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-api01-USER"}, clear=False):
            env = self._invoke(auth="api-key", token="sk-ant-oat01-FALLBACK")["env"]
        self.assertEqual(env["ANTHROPIC_API_KEY"], "sk-ant-api01-USER")
        self.assertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "sk-ant-oat01-FALLBACK")

    def test_bash_timeouts_pinned_to_run_budget(self):
        # Long foreground commands must stay synchronous: both Bash timeouts
        # are pinned to the engineer's whole `claude -p` budget (default
        # timeout_s = 3600s → 3_600_000 ms) so Claude Code never auto-moves a
        # training command to a background task mid-run.
        env = self._invoke()["env"]
        self.assertEqual(env["BASH_DEFAULT_TIMEOUT_MS"], str(60 * 60 * 1000))
        self.assertEqual(env["BASH_MAX_TIMEOUT_MS"], str(60 * 60 * 1000))


# ---------------------------------------------------------------------------
# 3) PreToolUse hook — boundary path checks
# ---------------------------------------------------------------------------

class HookTaskOutputAllowanceTests(unittest.TestCase):
    """The boundary hook denies reads outside the engineer's cwd, EXCEPT for
    Claude Code's own background-task output files for THIS run — otherwise an
    auto-backgrounded command leaves the agent blind to its own output. The
    allowance is scoped by the cwd slug embedded in the task path, so a
    sibling challenge's task output stays denied.
    """

    def _check(self, path: str, boundary: str, cwd: str) -> str | None:
        with mock.patch.dict(os.environ, {"TESTBED_BOUNDARY": boundary}, clear=False), \
             mock.patch("os.getcwd", return_value=cwd):
            return hook._path_violates_boundary(path)

    def test_allows_own_background_task_output(self):
        cwd = "/runs/v0/image_classification/aerial-cactus"
        slug = cwd.replace("/", "-")
        path = f"/private/tmp/claude-501/{slug}/9a78-uuid/tasks/btph5a74.output"
        self.assertIsNone(self._check(path, boundary=cwd, cwd=cwd))

    def test_denies_sibling_challenge_task_output(self):
        cwd = "/runs/v0/image_classification/aerial-cactus"
        other = "/runs/v0/image_classification/some-other-challenge"
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
        self.assertIsNone(self._check(f"{cwd}/submission/submission.csv",
                                      boundary=cwd, cwd=cwd))


if __name__ == "__main__":
    unittest.main()
