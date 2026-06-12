"""Tests for the remote-only / single-interface enforcement added to the
PreToolUse hook, the agent-prompt mode gating, and the auth-error retry
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
    "TESTBED_PLATFORM": "hopsworks",
    "TESTBED_COMPUTE_DENY": "torch,tensorflow,sklearn,xgboost",
}


class EnforceHookTests(unittest.TestCase):
    def _enforce(self, interface, tool, command=None, **input_extra):
        env = dict(_MARKERS, TESTBED_INTERFACE=interface)
        # Also clear TESTBED_CLI_SUBCOMMAND so a subcommand-entrypoint test can't
        # leak into the default (single-token) cases.
        for k in ("TESTBED_INTERFACE", "TESTBED_CLI_SUBCOMMAND", *_MARKERS):
            os.environ.pop(k, None)
        os.environ.update(env)
        tool_input = dict(input_extra)
        if command is not None:
            tool_input["command"] = command
        return hook.enforce(tool, tool_input)

    def _enforce_entrypoint(self, command, cli_binary="aws", cli_subcommand="sagemaker"):
        """Enforce in CLI mode with a SUBCOMMAND ENTRYPOINT (e.g. `aws sagemaker`):
        only `<cli_binary> <cli_subcommand> …` is on-interface."""
        for k in ("TESTBED_INTERFACE", "TESTBED_CLI_SUBCOMMAND", *_MARKERS):
            os.environ.pop(k, None)
        os.environ.update({
            "TESTBED_INTERFACE": "cli",
            "TESTBED_CLI_BINARY": cli_binary,
            "TESTBED_CLI_SUBCOMMAND": cli_subcommand,
            "TESTBED_SDK_MODULE": "sagemaker",
            "TESTBED_PLATFORM": "sagemaker",
            "TESTBED_COMPUTE_DENY": "torch,tensorflow,sklearn,xgboost",
        })
        return hook.enforce("Bash", {"command": command})

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

    def test_mcp_native_sdk_blocked(self):
        # The agent must NOT drive the SDK natively (locally) in MCP mode —
        # only the MCP tools may touch the platform. (Shipping SDK code to a
        # remote Job is server-side and uncounted; using it locally is the escape.)
        self.assertIsNotNone(
            self._enforce("mcp", "Bash", 'python -c "import hopsworks; hopsworks.login().get_feature_store()"')
        )

    def test_cli_native_sdk_blocked(self):
        # Likewise in CLI mode: only `hops`, never the SDK driven locally.
        self.assertIsNotNone(
            self._enforce("cli", "Bash", 'python -c "import hopsworks; fs.create_feature_group()"')
        )

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

    # --- fail-closed allowlist: only the interface + basic shell; everything
    #     else (node/ruby/curl/stray binaries) denied by default in EVERY mode ---
    def test_node_blocked_in_every_mode(self):
        # The field failure: when python was blocked the agent used `node -e`
        # to wrangle data locally. The allowlist denies it in all three modes.
        for iface in ("cli", "mcp", "sdk"):
            self.assertIsNotNone(self._enforce(iface, "Bash", 'node -e "console.log(1)"'), iface)

    def test_other_interpreters_blocked(self):
        for cmd in ('ruby -e "puts 1"', "perl -e 'print 1'", 'php -r "echo 1;"', "deno run x.ts"):
            self.assertIsNotNone(self._enforce("mcp", "Bash", cmd), cmd)

    def test_network_tools_blocked_in_every_mode(self):
        # curl/wget would hit the REST API directly, bypassing the interface
        # (and the api_calls log). Denied by the allowlist — not on it.
        for iface in ("cli", "mcp", "sdk"):
            self.assertIsNotNone(self._enforce(iface, "Bash", "curl https://host/api"), iface)
            self.assertIsNotNone(self._enforce(iface, "Bash", "wget https://host/x"), iface)

    def test_unknown_binary_blocked(self):
        self.assertIsNotNone(self._enforce("mcp", "Bash", "./some_custom_tool --go"))

    def test_sleep_blocked_in_every_mode(self):
        # `sleep` only stalls / burns compute budget; it does no interface work,
        # so it is off the allowlist and denied (incl. inside compound commands).
        for iface in ("cli", "mcp", "sdk"):
            self.assertIsNotNone(self._enforce(iface, "Bash", "sleep 30"), iface)
            self.assertIsNotNone(self._enforce(iface, "Bash", "sleep 5 && ls"), iface)

    def test_bash_c_node_blocked(self):
        # `bash -c "node …"` — the wrapper is unwrapped and the real exec gated.
        self.assertIsNotNone(self._enforce("cli", "Bash", 'bash -c "node -e \\"x\\""'))

    def test_which_node_allowed(self):
        # Probing availability is fine — the exec is `which`, not node.
        self.assertIsNone(self._enforce("mcp", "Bash", "which ruby node perl jq"))

    def test_basic_shell_inspection_allowed_in_mcp(self):
        for cmd in ("cat data/train.csv", "head -20 data/train.csv", "wc -l data/train.csv", "ls -la data/"):
            self.assertIsNone(self._enforce("mcp", "Bash", cmd), cmd)

    def test_pipeline_of_basic_shell_allowed(self):
        self.assertIsNone(self._enforce("mcp", "Bash", "cat data/train.csv | grep -c , | sort"))

    def test_redirect_not_misparsed_as_off_interface(self):
        # `2>&1` and `> out` must not split into a bogus segment that gets denied.
        self.assertIsNone(self._enforce("mcp", "Bash", "cp a b 2>&1"))
        self.assertIsNone(self._enforce("cli", "Bash", "cat data/train.csv > /tmp/out.txt"))

    def test_line_continuation_keeps_one_segment(self):
        # A multi-line `hops` command joined with backslash-newline must stay one
        # segment — the `\<nl>` is a continuation, not a separator. Otherwise the
        # continuation lines (`--primary-key id`) become bogus segments whose
        # first token is treated as a stray off-interface binary and denied.
        cmd = 'hops fg create --name trips \\\n  --primary-key id \\\n  --description "trip data"'
        self.assertEqual(len(hook._segments(cmd)), 1)
        self.assertIsNone(self._enforce("cli", "Bash", cmd))

    def test_line_continuation_real_separator_still_splits(self):
        # Folding `\<nl>` must not swallow a genuine separator on the next line:
        # a backgrounded interpreter after the continuation is still denied.
        cmd = 'hops fg create --name t \\\n  --primary-key id\nnode evil.js'
        self.assertIsNotNone(self._enforce("cli", "Bash", cmd))

    def test_escaped_char_not_treated_as_separator(self):
        # An escaped `;` is a literal char in the argument, not a command
        # separator — the single `hops` segment stays allowed in CLI mode.
        self.assertIsNone(self._enforce("cli", "Bash", "hops fg create --filter a\\;b"))

    # --- subcommand entrypoint (`aws sagemaker`): only that service is on-interface ---
    def test_entrypoint_subcommand_allowed(self):
        self.assertIsNone(self._enforce_entrypoint("aws sagemaker list-models"))
        # flags after the service are fine (still starts `aws sagemaker`)
        self.assertIsNone(self._enforce_entrypoint("aws sagemaker list-endpoints --max-results 5"))

    def test_entrypoint_other_service_denied(self):
        # Same binary, different service → off-interface escape.
        self.assertIsNotNone(self._enforce_entrypoint("aws s3 ls"))
        self.assertIsNotNone(self._enforce_entrypoint("aws ec2 describe-instances"))

    def test_entrypoint_bare_binary_denied(self):
        # `aws` with no service is not the `aws sagemaker` entrypoint.
        self.assertIsNotNone(self._enforce_entrypoint("aws configure list"))

    def test_entrypoint_subcommand_in_pipeline_allowed(self):
        # On-interface even piped into a basic shell util for inspection.
        self.assertIsNone(
            self._enforce_entrypoint("aws sagemaker list-models --output json | head -50")
        )

    def test_entrypoint_other_service_after_separator_denied(self):
        # `aws sagemaker …; aws s3 …` — the second segment is off-interface.
        self.assertIsNotNone(
            self._enforce_entrypoint("aws sagemaker list-models; aws s3 cp x y")
        )

    def test_entrypoint_allowlist_multiple_services(self):
        # `cli_subcommand` is an allowlist (comma-joined in the env): sagemaker
        # needs its S3 data plane and the runtime for endpoint invocation.
        subs = "sagemaker,sagemaker-runtime,s3"
        self.assertIsNone(self._enforce_entrypoint(
            "aws s3 cp data/train s3://bkt/train --recursive", cli_subcommand=subs))
        self.assertIsNone(self._enforce_entrypoint(
            "aws sagemaker-runtime invoke-endpoint --endpoint-name e out.json",
            cli_subcommand=subs))
        self.assertIsNone(self._enforce_entrypoint(
            "aws sagemaker list-training-jobs", cli_subcommand=subs))

    def test_entrypoint_allowlist_other_services_still_denied(self):
        subs = "sagemaker,sagemaker-runtime,s3"
        # s3api is a DIFFERENT service token than s3 — not on the allowlist.
        for cmd in ("aws ec2 describe-instances", "aws iam create-role --role-name x",
                    "aws s3api create-bucket --bucket x", "aws configure list"):
            msg = self._enforce_entrypoint(cmd, cli_subcommand=subs)
            self.assertIsNotNone(msg, cmd)
        # The denial names the full allowlist so the agent knows its options.
        self.assertIn("sagemaker-runtime", msg)

    def test_entrypoint_global_options_before_service_allowed(self):
        # The AWS CLI accepts global options BEFORE the service; the service
        # token after them is still the entrypoint, not the option.
        self.assertIsNone(self._enforce_entrypoint(
            "aws --region us-east-1 sagemaker list-models"))
        self.assertIsNone(self._enforce_entrypoint(
            "aws --output json --region us-east-1 sagemaker list-training-jobs"))
        self.assertIsNone(self._enforce_entrypoint(
            "aws --region=us-east-1 sagemaker list-models"))   # --opt=value form
        self.assertIsNone(self._enforce_entrypoint(
            "aws --debug sagemaker list-models"))              # valueless flag
        self.assertIsNone(self._enforce_entrypoint(
            "aws --no-cli-pager sagemaker list-models"))       # --no-* flag

    def test_entrypoint_global_options_other_service_still_denied(self):
        # Skipping options must not skip PAST an off-interface service.
        self.assertIsNotNone(self._enforce_entrypoint(
            "aws --region us-east-1 ec2 describe-instances"))
        self.assertIsNotNone(self._enforce_entrypoint(
            "aws --debug s3api create-bucket --bucket x"))

    def test_entrypoint_option_value_matching_service_not_entrypoint(self):
        # Fail closed: an option VALUE that happens to equal an allowed service
        # must not legitimize the real (off-interface) service after it.
        self.assertIsNotNone(self._enforce_entrypoint(
            "aws --profile sagemaker ec2 describe-instances"))

    def test_denials_logged_structurally(self):
        # A denied call must land in TESTBED_COMMAND_LOG with `denied: true` +
        # the reason — results.denied_calls counts these records instead of
        # substring-scanning transcripts.
        import io
        import json
        from unittest import mock
        log = Path(tempfile.mkdtemp()) / "commands.jsonl"
        for k in ("TESTBED_INTERFACE", "TESTBED_CLI_SUBCOMMAND", *_MARKERS):
            os.environ.pop(k, None)
        os.environ.update(dict(_MARKERS, TESTBED_INTERFACE="cli",
                               TESTBED_COMMAND_LOG=str(log)))
        try:
            payload = {"tool_name": "Bash",
                       "tool_input": {"command": 'python -c "import torch"'}}
            with mock.patch("sys.stdin", io.StringIO(json.dumps(payload))):
                rc = hook.main()
            self.assertEqual(rc, 2)
            allowed = {"tool_name": "Bash", "tool_input": {"command": "hops fg list"}}
            with mock.patch("sys.stdin", io.StringIO(json.dumps(allowed))):
                rc = hook.main()
            self.assertEqual(rc, 0)
            recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
            self.assertEqual(len(recs), 2)
            self.assertTrue(recs[0].get("denied"))
            self.assertIn("DENIED:", recs[0]["reason"])
            self.assertNotIn("denied", recs[1])
        finally:
            os.environ.pop("TESTBED_COMMAND_LOG", None)

    # --- denial messages name the ACTIVE platform/interface, never a hardcoded one ---
    def test_denial_message_names_active_interface(self):
        # Regression: messages used to hardcode Hopsworks ("Use only the `hops`
        # CLI") regardless of platform — a sagemaker agent denied `aws s3`
        # was told to use `hops`. Must derive from TESTBED_* env instead.
        msg = self._enforce_entrypoint("aws s3 ls")
        self.assertIn("aws sagemaker", msg)
        self.assertNotIn("hops", msg)
        msg = self._enforce_entrypoint('python -c "import torch"')
        self.assertIn("sagemaker", msg)
        self.assertNotIn("Hopsworks", msg)

    def test_denial_message_falls_back_without_platform_env(self):
        # No TESTBED_PLATFORM (e.g. an old run dir's settings) → generic wording,
        # still no hardcoded platform.
        env = {k: v for k, v in _MARKERS.items() if k != "TESTBED_PLATFORM"}
        for k in ("TESTBED_INTERFACE", "TESTBED_CLI_SUBCOMMAND", *_MARKERS):
            os.environ.pop(k, None)
        os.environ.update(dict(env, TESTBED_INTERFACE="mcp"))
        msg = hook.enforce("Bash", {"command": "hops project use x"})
        self.assertIn("the platform's MCP tools", msg)
        msg = hook.enforce("Bash", {"command": 'python -c "print(1)"'})
        self.assertIn("the remote platform", msg)

    def test_mcp_floor_submission_allowed(self):
        self.assertIsNone(
            self._enforce("mcp", "Bash", "mkdir -p submission && cp data/sample_submission.csv submission/submission.csv")
        )

    def test_compound_keywords_not_denied(self):
        # Shell control-flow keywords are skipped, not treated as binaries.
        self.assertIsNone(
            self._enforce("cli", "Bash", 'for f in a b; do cp "$f" out/; done')
        )

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
    #     before the run, so the agent never needs pip there. ---
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


class InstanceTypeGuardTests(unittest.TestCase):
    """Free-tier guard: with TESTBED_INSTANCE_ALLOW set, any `ml.<family>.<size>`
    token outside the allowlist is denied — in Bash commands, executed python
    payloads, MCP tool args, and Write/Edit content."""

    FREE_TIER = "ml.t3.medium,ml.m4.xlarge,ml.m5.xlarge"

    def _check(self, tool, tool_input, allow=FREE_TIER):
        os.environ.pop("TESTBED_INSTANCE_ALLOW", None)
        if allow is not None:
            os.environ["TESTBED_INSTANCE_ALLOW"] = allow
        try:
            return hook.enforce_instance_types(tool, tool_input)
        finally:
            os.environ.pop("TESTBED_INSTANCE_ALLOW", None)

    def test_no_allowlist_no_enforcement(self):
        self.assertIsNone(self._check(
            "Bash", {"command": "aws sagemaker create-training-job --resource-config "
                                "InstanceType=ml.p3.2xlarge,InstanceCount=1"},
            allow=None))

    def test_cli_free_tier_instance_allowed(self):
        self.assertIsNone(self._check(
            "Bash", {"command": "aws sagemaker create-training-job --resource-config "
                                "InstanceType=ml.m5.xlarge,InstanceCount=1,VolumeSizeInGB=10"}))

    def test_cli_gpu_instance_denied(self):
        reason = self._check(
            "Bash", {"command": "aws sagemaker create-training-job --resource-config "
                                "InstanceType=ml.p3.2xlarge,InstanceCount=1"})
        self.assertIsNotNone(reason)
        self.assertIn("ml.p3.2xlarge", reason)

    def test_cli_big_cpu_instance_denied(self):
        self.assertIsNotNone(self._check(
            "Bash", {"command": "aws sagemaker create-endpoint-config --production-variants "
                                "VariantName=v1,InstanceType=ml.m5.24xlarge,InitialInstanceCount=1"}))

    def test_mcp_args_denied(self):
        self.assertIsNotNone(self._check(
            "mcp__sagemaker__create_training_job",
            {"resource_config": {"InstanceType": "ml.g5.xlarge", "InstanceCount": 1}}))

    def test_mcp_args_free_tier_allowed(self):
        self.assertIsNone(self._check(
            "mcp__sagemaker__create_training_job",
            {"resource_config": {"InstanceType": "ml.m4.xlarge", "InstanceCount": 1}}))

    def test_write_job_spec_denied(self):
        # A job spec written to disk first (`--cli-input-json file://job.json`)
        # is caught at Write time.
        self.assertIsNotNone(self._check(
            "Write", {"file_path": "/x/job.json",
                      "content": '{"ResourceConfig": {"InstanceType": "ml.trn1.32xlarge"}}'}))

    def test_executed_script_payload_denied(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "train.py"
            script.write_text("est = Estimator(instance_type='ml.c5.18xlarge')\n")
            cwd = os.getcwd()
            os.chdir(td)
            try:
                self.assertIsNotNone(self._check("Bash", {"command": "python train.py"}))
            finally:
                os.chdir(cwd)

    def test_serverless_no_instance_type_allowed(self):
        self.assertIsNone(self._check(
            "Bash", {"command": "aws sagemaker create-endpoint-config --production-variants "
                                "VariantName=v1,ServerlessConfig={MemorySizeInMB=2048,MaxConcurrency=1}"}))

    def test_plain_text_not_misflagged(self):
        # `html.parser` etc. must not match the ml.<family>.<size> pattern.
        self.assertIsNone(self._check(
            "Bash", {"command": 'grep "html.parser" data/description.md'}))


class BanterRunForegroundTests(unittest.TestCase):
    """A nested `banter run` must be foreground —
    never piped (SIGPIPE-kills it mid-build) or backgrounded."""

    def _misuse(self, command):
        return hook._banter_run_misuse(command)

    # --- blocked: piping / backgrounding ---
    def test_pipe_to_head_blocked(self):
        # This is the exact failure from the field: `… 2>&1 | head -300`.
        self.assertIsNotNone(
            self._misuse("banter run --category t --task c --platform hopsworks 2>&1 | head -300")
        )

    def test_pipe_to_tail_blocked(self):
        self.assertIsNotNone(self._misuse("banter run --task t | tail -40"))

    def test_absolute_banter_path_pipe_blocked(self):
        self.assertIsNotNone(
            self._misuse("/Users/x/testbed/.venv/bin/banter run --task t | head -5")
        )

    def test_background_ampersand_blocked(self):
        self.assertIsNotNone(self._misuse("banter run --category t --task c &"))

    def test_background_then_poll_blocked(self):
        self.assertIsNotNone(self._misuse("banter run --task t & sleep 30"))

    def test_pipe_after_cd_guard_blocked(self):
        # The cd-guard prefix is fine; the trailing pipe on `banter run` is not.
        self.assertIsNotNone(
            self._misuse('cd /run && banter run --category t --task c | head -100')
        )

    # --- allowed: foreground, redirects, other subcommands ---
    def test_plain_foreground_allowed(self):
        self.assertIsNone(self._misuse("banter run --category t --task c --platform hopsworks"))

    def test_redirect_to_file_allowed(self):
        # The sanctioned way to cap output: redirect, then read agent.log.
        self.assertIsNone(self._misuse("banter run --task t > run.log 2>&1"))

    def test_redirect_to_devnull_allowed(self):
        self.assertIsNone(self._misuse("banter run --category t --task c > /dev/null 2>&1"))

    def test_redirect_then_chained_tail_allowed(self):
        # `&&` chains a SEPARATE tail of agent.log — not a pipe of banter run.
        self.assertIsNone(
            self._misuse("banter run --task t > run.log 2>&1 && tail -60 v1/t/c/agent.log")
        )

    def test_budget_check_piped_not_blocked(self):
        # Only `banter run` is gated; other subcommands may be piped freely.
        self.assertIsNone(self._misuse("banter budget-check --start 1 | grep CONTINUE"))

    def test_non_banter_pipe_allowed(self):
        self.assertIsNone(self._misuse("ls v1 | head -5"))

    def test_2to1_redirect_alone_not_flagged_as_background(self):
        # `2>&1` without a pipe/`&` must NOT be misread as backgrounding.
        self.assertIsNone(self._misuse("banter run --category t --task c 2>&1"))


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
        self.assertIn("you are the LOCAL BASELINE", text)
        self.assertNotIn("nothing runs locally", text.lower())
        self.assertNotIn("The interface is what's being measured", text)
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


class RateLimitWaitAccountingTests(unittest.TestCase):
    """`run_with_retry` returns the back-off sleep total as a third value so
    callers can report compute time (wall − wait) — results rows must not be
    penalized for time spent waiting on rate limits."""

    def test_backoff_wait_returned_separately_from_wall(self):
        import json
        from unittest import mock
        d = Path(tempfile.mkdtemp())
        marker = d / "ran_once"
        rl = json.dumps({"type": "result", "is_error": True,
                         "api_error_status": "429", "result": "rate_limit"})
        ok = json.dumps({"type": "result", "is_error": False, "result": "ok"})
        # First attempt: rate-limited result + non-zero exit → one back-off
        # sleep (base 2s, mocked). Second attempt: success.
        script = (f"if [ -e {marker} ]; then echo '{ok}'; "
                  f"else touch {marker}; echo '{rl}'; exit 1; fi")
        with mock.patch("time.sleep") as slept:
            exit_code, wall, wait = claude_runner.run_with_retry(
                cmd=["sh", "-c", script],
                cwd=d,
                env={},
                transcript_path=d / "transcript.jsonl",
                stderr_path=d / "stderr.log",
            )
        self.assertEqual(exit_code, 0)
        self.assertEqual(wait, claude_runner.RATE_LIMIT_BASE_BACKOFF_S)
        slept.assert_called_once_with(claude_runner.RATE_LIMIT_BASE_BACKOFF_S)
        # Wall stays the true elapsed time; the wait is reported alongside, not
        # silently folded in or subtracted here (callers decide).
        self.assertGreaterEqual(wall, 0.0)


class DeadRowSchemaTests(unittest.TestCase):
    def test_error_column_present_and_last_is_run_dir(self):
        from banter import results
        self.assertIn("error", results.RESULTS_FIELDS)
        self.assertEqual(results.RESULTS_FIELDS[-1], "run_dir")  # invariant preserved

    def test_dead_row_round_trips_with_error_and_zeros(self):
        from banter import results
        import csv
        out = Path(tempfile.mkdtemp()) / "results.csv"
        row = results.Row(
            started_at="2026-06-04T00:00:00+00:00", run="9", version="v1",
            platform="hopsworks", interface="sdk", skills="none",
            category="t", task="c", sdk_calls=0,
            error="no valid submission produced", run_dir=str(out.parent),
        )
        results.append(out, row)
        got = list(csv.DictReader(out.open()))[0]
        self.assertEqual(got["error"], "no valid submission produced")
        self.assertEqual(got["asserts_passed"], "0")  # Row default
        self.assertEqual(got["asserts_total"], "0")


if __name__ == "__main__":
    unittest.main()
