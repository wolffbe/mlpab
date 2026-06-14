"""Tests for the engine-agnostic in-flight exec gate (hooks/exec_gate.py) that
gives vibe/codex the SAME single-interface enforcement as the Claude PreToolUse
hook. Includes an end-to-end check that mirrors how vibe invokes the shell
(`create_subprocess_shell(cmd, executable=$SHELL)`)."""

import asyncio
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mlpab import claude_runner

_GATE = Path(__file__).resolve().parents[1] / "src" / "mlpab" / "hooks" / "exec_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("exec_gate", _GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ExtractCommandTests(unittest.TestCase):
    def setUp(self):
        self.gate = _load_gate()

    def test_dash_c(self):
        self.assertEqual(self.gate._extract_command(["-c", "ls -la"]), "ls -la")

    def test_combined_login_c(self):
        # vibe/codex login-shell form: `-lc "<cmd>"`.
        self.assertEqual(self.gate._extract_command(["-lc", "aws s3 ls"]), "aws s3 ls")

    def test_split_login_then_c(self):
        self.assertEqual(self.gate._extract_command(["-l", "-c", "echo hi"]), "echo hi")

    def test_no_command_returns_none(self):
        self.assertIsNone(self.gate._extract_command(["-l"]))
        self.assertIsNone(self.gate._extract_command([]))

    def test_dash_dash_before_command(self):
        # `-c -- CMD`: the end-of-options `--` must be skipped, not taken as the
        # command (else the real CMD runs ungated).
        self.assertEqual(self.gate._extract_command(["-c", "--", "node -e 1"]), "node -e 1")
        self.assertEqual(self.gate._extract_command(["-lc", "--", "ls"]), "ls")

    def test_unknown_flag_cluster_not_treated_as_c(self):
        # A token like `-Zc` (Z not a shell flag letter) must NOT be read as -c.
        self.assertIsNone(self.gate._extract_command(["-Zc", "ls"]))


class GateSubprocessTests(unittest.TestCase):
    """Run the gate as a real subprocess (the way an engine invokes $SHELL):
    `python exec_gate.py -c <cmd>` with the TESTBED_* enforcement env set."""

    def _env(self, **overrides):
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("TESTBED_") and k != "MLPAB_REAL_SHELL"
        }
        env["TESTBED_INTERFACE"] = "cli"
        env["TESTBED_CLI_BINARY"] = "aws"
        env["TESTBED_CLI_SUBCOMMAND"] = "sagemaker,s3"
        env["TESTBED_COMPUTE_DENY"] = "torch,sklearn"
        env["MLPAB_REAL_SHELL"] = "/bin/bash"
        env.update(overrides)
        return env

    def _run(self, command, **env_overrides):
        return subprocess.run(
            [sys.executable, str(_GATE), "-c", command],
            env=self._env(**env_overrides),
            capture_output=True,
            text=True,
        )

    def test_allowed_interface_command_runs(self):
        # An on-interface-ish command that's really just echo (no network): the
        # gate should pass it through to the real shell and it executes.
        r = self._run("echo ok")
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout.strip(), "ok")

    def test_offinterface_python_denied(self):
        r = self._run('python -c "import torch"')
        self.assertEqual(r.returncode, 2)
        self.assertIn("DENIED", r.stderr)
        # The denied command must NOT have executed.
        self.assertNotIn("torch", r.stdout)

    def test_solution_answer_key_denied(self):
        r = self._run("cat ../solution/truth.json")
        self.assertEqual(r.returncode, 2)
        self.assertIn("answer key", r.stderr)

    def test_denial_writes_structured_commands_log(self):
        # The gate must record denials as `denied: true` in TESTBED_COMMAND_LOG so
        # results.denied_calls counts vibe/codex denials structurally.
        import json as _json

        log = Path(tempfile.mkdtemp()) / "commands.jsonl"
        self._run('python -c "import torch"', TESTBED_COMMAND_LOG=str(log))
        recs = [_json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        self.assertTrue(recs and recs[-1].get("denied") is True)
        self.assertIn("DENIED", recs[-1].get("reason", ""))

    def test_command_substitution_escape_denied(self):
        r = self._run('echo $(node -e "1")')
        self.assertEqual(r.returncode, 2)
        self.assertIn("DENIED", r.stderr)

    def test_none_interface_passes_through(self):
        # No TESTBED_INTERFACE → baseline run, gate enforces nothing: a command
        # that WOULD be denied under cli mode runs through to the real shell.
        r = self._run(f'{sys.executable} -c "print(123)"', TESTBED_INTERFACE="")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("123", r.stdout)


class VibeStyleEndToEndTest(unittest.TestCase):
    """Exactly how vibe spawns commands: create_subprocess_shell(cmd,
    executable=<gate>). Proves the $SHELL wiring blocks an off-interface
    command before it runs."""

    def test_executable_shell_blocks_offinterface(self):
        async def _go():
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith("TESTBED_")
            }
            env.update(
                {
                    "TESTBED_INTERFACE": "cli",
                    "TESTBED_CLI_BINARY": "aws",
                    "TESTBED_COMPUTE_DENY": "torch",
                    "MLPAB_REAL_SHELL": "/bin/bash",
                }
            )
            marker = Path(tempfile.mkdtemp()) / "ran"
            # The command would create the marker if it executed; the gate must
            # deny it (off-interface local python importing torch) first.
            cmd = f'python -c "import torch" ; touch {marker}'
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                executable=str(_GATE),
                env=env,
            )
            _out, err = await proc.communicate()
            return proc.returncode, err.decode(), marker.exists()

        rc, err, ran = asyncio.run(_go())
        self.assertEqual(rc, 2)
        self.assertIn("DENIED", err)
        self.assertFalse(ran, "denied command should not have executed")


class InstallExecGateTests(unittest.TestCase):
    def test_wires_shell_and_path_when_interface_set(self):
        run_dir = Path(tempfile.mkdtemp())
        env = {"TESTBED_INTERFACE": "cli", "PATH": "/usr/bin"}
        claude_runner.install_exec_gate(run_dir, env)
        gate_dir = run_dir.resolve() / ".mlpab_gate"
        self.assertEqual(env["SHELL"], str(gate_dir / "exec_gate.py"))
        self.assertTrue(env["PATH"].startswith(f"{gate_dir}{os.pathsep}"))
        self.assertTrue((gate_dir / "exec_gate.py").exists())
        self.assertTrue((gate_dir / "log_tool_call.py").exists())
        self.assertTrue((gate_dir / "bash").exists())
        self.assertEqual(env.get("MLPAB_REAL_SHELL"), "/bin/bash")

    def test_noop_for_baseline(self):
        run_dir = Path(tempfile.mkdtemp())
        env = {"PATH": "/usr/bin"}  # no TESTBED_INTERFACE
        claude_runner.install_exec_gate(run_dir, env)
        self.assertNotIn("SHELL", env)
        self.assertEqual(env["PATH"], "/usr/bin")
        self.assertFalse((run_dir / ".mlpab_gate").exists())


if __name__ == "__main__":
    unittest.main()
