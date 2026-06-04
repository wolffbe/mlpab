"""Tests for the remote-only / single-interface enforcement added to the
PreToolUse hook, the engineer-prompt mode gating, and the auth-error retry
detection in claude_runner."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from banter import claude_runner, runner

# The hook is designed to run standalone (copied into each run dir), so load it
# straight from its file — exactly how the harness executes it.
_HOOK_PATH = Path(__file__).resolve().parents[1] / "src" / "banter" / "hooks" / "log_tool_call.py"
_spec = importlib.util.spec_from_file_location("log_tool_call", _HOOK_PATH)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)

_MARKERS = {
    "TESTBED_SDK_MODULE": "hopsworks",
    "TESTBED_CLI_BINARY": "hops",
    "TESTBED_COMPUTE_DENY": "torch,tensorflow,sklearn,xgboost",
}


class EnforceHookTests(unittest.TestCase):
    def _enforce(self, interface, tool, command=None, **input_extra):
        env = dict(_MARKERS, TESTBED_INTERFACE=interface)
        for k in ("TESTBED_INTERFACE", *_MARKERS):
            os.environ.pop(k, None)
        os.environ.update(env)
        tool_input = dict(input_extra)
        if command is not None:
            tool_input["command"] = command
        return hook.enforce(tool, tool_input)

    # --- compute libraries: blocked locally in every mode ---
    def test_local_torch_blocked_in_cli(self):
        self.assertIsNotNone(self._enforce("cli", "Bash", 'python -c "import torch"'))

    def test_local_sklearn_comma_import_blocked_in_sdk(self):
        # `import hopsworks, sklearn` — the comma-list must still be caught.
        self.assertIsNotNone(
            self._enforce("sdk", "Bash", 'python -c "import hopsworks, sklearn; sklearn.fit()"')
        )

    def test_script_file_with_torch_blocked(self):
        d = Path(tempfile.mkdtemp())
        (d / "train.py").write_text("import torch\nmodel.fit()\n")
        cwd = os.getcwd()
        os.chdir(d)
        try:
            self.assertIsNotNone(self._enforce("sdk", "Bash", "python train.py"))
        finally:
            os.chdir(cwd)

    def test_pip_install_torch_allowed(self):
        # Installing is setup, not local execution of the library.
        self.assertIsNone(self._enforce("sdk", "Bash", "pip install torch"))

    def test_echo_import_torch_not_executed_allowed(self):
        self.assertIsNone(self._enforce("cli", "Bash", 'echo "import torch"'))

    # --- CLI mode: only hops ---
    def test_cli_hops_allowed(self):
        self.assertIsNone(self._enforce("cli", "Bash", "hops jobs create --name t"))

    def test_cli_any_local_python_blocked(self):
        self.assertIsNotNone(self._enforce("cli", "Bash", 'python -c "import pandas"'))

    def test_cli_cp_floor_submission_allowed(self):
        self.assertIsNone(
            self._enforce("cli", "Bash", "mkdir -p submission && cp data/sample_submission.csv submission/submission.csv")
        )

    def test_cli_mcp_tool_blocked(self):
        self.assertIsNotNone(self._enforce("cli", "mcp__hopsworks__create_job"))

    # --- MCP mode: only the tools ---
    def test_mcp_tool_allowed(self):
        self.assertIsNone(self._enforce("mcp", "mcp__hopsworks__create_job"))

    def test_mcp_local_python_blocked(self):
        self.assertIsNotNone(self._enforce("mcp", "Bash", 'python -c "print(1)"'))

    def test_mcp_hops_blocked(self):
        self.assertIsNotNone(self._enforce("mcp", "Bash", "hops project create x"))

    # --- SDK mode: only hopsworks python, no ML ---
    def test_sdk_hopsworks_python_allowed(self):
        self.assertIsNone(
            self._enforce("sdk", "Bash", 'python -c "import hopsworks; j=p.create_job(); j.run()"')
        )

    def test_sdk_pandas_glue_allowed(self):
        self.assertIsNone(self._enforce("sdk", "Bash", 'python -c "import pandas as pd; pd.read_csv(1)"'))

    def test_sdk_hops_blocked(self):
        self.assertIsNotNone(self._enforce("sdk", "Bash", "hops jobs run x"))

    def test_sdk_mcp_tool_blocked(self):
        self.assertIsNotNone(self._enforce("sdk", "mcp__hopsworks__x"))

    # --- parser-bypass regressions (separators, interpreters, dynamic import) ---
    def test_semicolon_glued_separator_blocked(self):
        # shlex keeps `true;python` as one token — the raw splitter must still
        # see the second command. (CLI mode: no local python.)
        self.assertIsNotNone(self._enforce("cli", "Bash", "true;python train.py"))

    def test_semicolon_attached_to_prev_token_blocked(self):
        self.assertIsNotNone(self._enforce("cli", "Bash", "cd foo; python train.py"))

    def test_versioned_interpreter_blocked(self):
        self.assertIsNotNone(self._enforce("cli", "Bash", "python3.11 train.py"))
        self.assertIsNotNone(self._enforce("cli", "Bash", "/usr/bin/python3.12 train.py"))

    def test_env_and_uv_prefixed_python_blocked(self):
        self.assertIsNotNone(self._enforce("cli", "Bash", "env X=1 python train.py"))
        self.assertIsNotNone(self._enforce("cli", "Bash", "uv run python train.py"))

    def test_dynamic_import_of_ml_lib_blocked(self):
        self.assertIsNotNone(self._enforce("sdk", "Bash", 'python -c "__import__(\'torch\')"'))
        self.assertIsNotNone(
            self._enforce("sdk", "Bash", 'python -c "import importlib; importlib.import_module(\'sklearn\')"')
        )

    # --- installs stay LOCKED in CLI/MCP: banter installs the interface + deps
    #     before the run, so the engineer never needs pip there. ---
    def test_pip_blocked_in_cli_and_mcp(self):
        for iface in ("cli", "mcp"):
            self.assertIsNotNone(self._enforce(iface, "Bash", "pip install pandas"), iface)
            self.assertIsNotNone(self._enforce(iface, "Bash", "python -m pip install pandas"), iface)

    def test_pip_allowed_in_sdk_mode(self):
        # SDK mode is python-allowed (the SDK IS python); pip is just tooling and
        # an installed ML lib still can't be EXECUTED (rule 1 blocks `import`).
        self.assertIsNone(self._enforce("sdk", "Bash", "pip install pandas"))
        self.assertIsNone(self._enforce("sdk", "Bash", "python -m pip install torch"))

    def test_sdk_unreadable_script_fails_closed(self):
        # SDK mode allows python, but an executed script we can't read to verify
        # it's ML-free must be blocked (fail closed), not allowed.
        self.assertIsNotNone(self._enforce("sdk", "Bash", "python /nonexistent/dir/train.py"))

    def test_sdk_inline_and_no_script_not_affected_by_failclosed(self):
        # `-c` and bare interpreter have no script file → no unreadable → allowed.
        self.assertIsNone(self._enforce("sdk", "Bash", 'python -c "import hopsworks; j.run()"'))

    def test_sdk_py_arg_not_treated_as_unreadable_script(self):
        # A trailing `.py` ARG (config) is not the executed script; with the real
        # script readable and ML-free, the command is allowed.
        d = Path(tempfile.mkdtemp())
        (d / "drive.py").write_text("import hopsworks\nhopsworks.login()\n")
        cwd = os.getcwd(); os.chdir(d)
        try:
            self.assertIsNone(self._enforce("sdk", "Bash", "python drive.py --config missing_cfg.py"))
        finally:
            os.chdir(cwd)

    def test_semicolon_inside_quotes_not_a_separator(self):
        # The `;` is inside the -c body, not a shell separator — single segment,
        # payload intact, and it imports only os/pandas → allowed in SDK mode.
        self.assertIsNone(self._enforce("sdk", "Bash", 'python -c "import os; os.getcwd()"'))

    # --- no interface under test → no enforcement (none/none baseline) ---
    def test_no_interface_no_enforcement(self):
        for k in ("TESTBED_INTERFACE", *_MARKERS):
            os.environ.pop(k, None)
        self.assertIsNone(hook.enforce("Bash", {"command": 'python -c "import torch"'}))

    def test_none_interface_trains_locally(self):
        # The none/none baseline (interface "none") must NOT be enforced — it
        # legitimately trains locally with torch.
        self.assertIsNone(self._enforce("none", "Bash", 'python -c "import torch; train()"'))


class PromptModeGatingTests(unittest.TestCase):
    def test_under_test_is_remote_only(self):
        text = runner._build_prompt("comp", "FRAG", interface_under_test=True)
        self.assertIn("nothing runs locally", text.lower())
        self.assertIn("HARD-ENFORCED", text)        # restrictions stated explicitly
        self.assertIn("Always BLOCKED", text)
        self.assertIn("give up", text.lower())
        self.assertNotIn("HF_HOME", text)           # local-only block stripped
        self.assertNotIn("UNDER_TEST", text)        # markers consumed
        self.assertNotIn("LOCAL_ONLY", text)

    def test_baseline_is_local_training(self):
        text = runner._build_prompt("comp", "FRAG", interface_under_test=False)
        self.assertIn("Build and train a model", text)
        self.assertNotIn("Everything runs on Hopsworks", text)
        self.assertNotIn("UNDER_TEST", text)
        self.assertNotIn("LOCAL_ONLY", text)


class AuthRetryDetectionTests(unittest.TestCase):
    def _transcript(self, result_event):
        import json
        p = Path(tempfile.mkdtemp()) / "transcript.jsonl"
        p.write_text(json.dumps(result_event) + "\n")
        return p

    def test_401_detected_as_auth_error(self):
        tr = self._transcript({"type": "result", "is_error": True,
                               "result": "API Error: 401 Invalid authentication credentials"})
        self.assertTrue(claude_runner._last_result_is_auth_error(tr))
        self.assertFalse(claude_runner._last_result_is_rate_limited(tr))

    def test_429_is_rate_limit_not_auth(self):
        tr = self._transcript({"type": "result", "is_error": True,
                               "api_error_status": "429", "result": "rate_limit"})
        self.assertTrue(claude_runner._last_result_is_rate_limited(tr))
        self.assertFalse(claude_runner._last_result_is_auth_error(tr))

    def test_success_is_neither(self):
        tr = self._transcript({"type": "result", "is_error": False, "result": "ok"})
        self.assertFalse(claude_runner._last_result_is_auth_error(tr))
        self.assertFalse(claude_runner._last_result_is_rate_limited(tr))


class LaunderingAuditTests(unittest.TestCase):
    def _src(self, files: dict):
        root = Path(tempfile.mkdtemp())
        for rel, text in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        return root

    def test_flags_mcp_tool_running_local_python(self):
        from banter import results
        src = self._src({
            "python/hopsworks/mcp/tools/train.py": "import subprocess\nsubprocess.run(['python','t.py'])\n",
        })
        flagged = results.audit_interface_local_exec(src)
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["file"], "python/hopsworks/mcp/tools/train.py")
        self.assertIn("subprocess", flagged[0]["patterns"])

    def test_clean_tool_not_flagged(self):
        from banter import results
        src = self._src({
            "python/hopsworks/mcp/tools/jobs.py": "def run_job(c):\n    return client.post('/jobs')\n",
        })
        self.assertEqual(results.audit_interface_local_exec(src), [])

    def test_non_entrypoint_subprocess_ignored(self):
        from banter import results
        # subprocess in a non-tool/command file is not the laundering vector.
        src = self._src({
            "python/hopsworks/core/util.py": "import subprocess\nsubprocess.run(['x'])\n",
        })
        self.assertEqual(results.audit_interface_local_exec(src), [])

    def test_unchanged_upstream_not_flagged_against_baseline(self):
        from banter import results
        body = "import subprocess\nsubprocess.run(['x'])\n"
        base = self._src({"python/hopsworks/cli/commands/job.py": body})
        cur = self._src({"python/hopsworks/cli/commands/job.py": body})
        # Identical to baseline → researcher didn't introduce it → not flagged.
        self.assertEqual(results.audit_interface_local_exec(cur, base), [])

    def test_researcher_added_exec_flagged_against_baseline(self):
        from banter import results
        base = self._src({"python/hopsworks/cli/commands/job.py": "def run(): pass\n"})
        cur = self._src({"python/hopsworks/cli/commands/job.py": "import subprocess\ndef run(): subprocess.run(['x'])\n"})
        flagged = results.audit_interface_local_exec(cur, base)
        self.assertEqual(len(flagged), 1)

    def test_sdk_source_laundering_flagged_with_baseline(self):
        from banter import results
        # An SDK-interface file (not under mcp/tools or cli/commands): only
        # scanned when a baseline is present, and only if the researcher ADDED
        # the local-exec pattern.
        rel = "python/hopsworks/core/engine.py"
        base = self._src({rel: "def train(): return remote_job()\n"})
        cur = self._src({rel: "import subprocess\ndef train(): subprocess.run(['python','t.py'])\n"})
        self.assertEqual(len(results.audit_interface_local_exec(cur, base)), 1)
        # Without a baseline, a non-entrypoint file is NOT scanned (avoids
        # flagging legitimate upstream subprocess use).
        self.assertEqual(results.audit_interface_local_exec(cur), [])

    def test_preexisting_subprocess_unchanged_pattern_not_flagged(self):
        from banter import results
        # Researcher edits a file that ALREADY used subprocess, without adding a
        # new local-exec pattern → not flagged (the pattern isn't researcher-new).
        rel = "python/hopsworks/cli/commands/job.py"
        base = self._src({rel: "import subprocess\ndef run(): subprocess.run(['x'])\n"})
        cur = self._src({rel: "import subprocess\ndef run(): subprocess.run(['x'])  # tweaked comment\n"})
        self.assertEqual(results.audit_interface_local_exec(cur, base), [])


class DeadRowSchemaTests(unittest.TestCase):
    def test_error_column_present_and_last_is_run_dir(self):
        from banter import results
        self.assertIn("error", results.FIELDS)
        self.assertEqual(results.FIELDS[-1], "run_dir")  # invariant preserved

    def test_dead_row_round_trips_with_error_and_zeros(self):
        from banter import results
        import csv
        out = Path(tempfile.mkdtemp()) / "results.csv"
        row = results.Row(
            started_at="2026-06-04T00:00:00+00:00", run="9", version="v1",
            platform="hopsworks", interface="sdk", skills="none",
            prev_run="", prev_version="", task="t", challenge="c",
            valid_submission=0, score=0.0, sdk_calls=0,
            error="no valid submission produced", run_dir=str(out.parent),
        )
        results.append(out, row)
        got = list(csv.DictReader(out.open()))[0]
        self.assertEqual(got["error"], "no valid submission produced")
        self.assertEqual(got["valid_submission"], "0")
        self.assertEqual(float(got["score"]), 0.0)


if __name__ == "__main__":
    unittest.main()
