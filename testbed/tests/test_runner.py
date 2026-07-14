"""Tests for runner helpers: dynamic environment detection + prompt rendering."""

import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from mlpab import evals_provider, interfaces, runner


class SeedForTests(unittest.TestCase):
    def test_deterministic_and_distinct_per_attempt(self):
        a = evals_provider.seed_for("rq1", "feature", "training_data", 1)
        self.assertEqual(a, evals_provider.seed_for("rq1", "feature", "training_data", 1))
        self.assertNotEqual(a, evals_provider.seed_for("rq1", "feature", "training_data", 2))
        # Non-negative 31-bit int (usable as a numpy/random seed).
        self.assertGreaterEqual(a, 0)
        self.assertLess(a, 2**31)

    def test_unimplemented_family_fails_fast(self):
        with self.assertRaises(ValueError):
            evals_provider._family("not-a-real-eval")


class PrepareReseedTests(unittest.TestCase):
    """prepare() deterministically reseeds when a generator's validity gate
    rejects a seed, instead of letting the GateError kill the combo forever
    (the seed is fixed per config, so a bad seed would fail on every run)."""

    @staticmethod
    def _fake_generator(fail_times, seen):
        """A stand-in evals.<fam>.generate: raises its own GateError for the
        first `fail_times` seeds, then stages a minimal valid instance."""

        class _FakeGate(RuntimeError):
            pass

        mod = types.SimpleNamespace(GateError=_FakeGate)
        state = {"calls": 0}

        def generate(seed, staging):
            seen.append(seed)
            if state["calls"] < fail_times:
                state["calls"] += 1
                raise _FakeGate(f"gate reject seed={seed}")
            staging = Path(staging)
            (staging / "data").mkdir(parents=True)
            (staging / "data" / "rows.csv").write_text("a\n1\n")
            (staging / "solution").mkdir(parents=True)
            (staging / "solution" / "truth.json").write_text("{}")
            (staging / "prompt.txt").write_text("BODY")
            (staging / "instance.json").write_text(json.dumps({"seed": seed}))

        mod.generate = generate
        return mod

    def _prepare_with(self, fake, task, run_dir, seed):
        with (
            mock.patch.object(evals_provider, "_family", return_value=("fake.fam", "table")),
            mock.patch("importlib.import_module", return_value=fake),
        ):
            return evals_provider.prepare(task, run_dir, seed)

    def _attempt_dir(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, tmp, True)
        run_dir = tmp / "1" / "task"
        run_dir.mkdir(parents=True)
        return run_dir

    def test_passing_seed_unchanged_first_try(self):
        seen = []
        run_dir = self._attempt_dir()
        body = self._prepare_with(self._fake_generator(0, seen), "anytask", run_dir, 4242)
        self.assertEqual(body, "BODY")
        self.assertEqual(seen, [4242])  # no reseed: succeeded on the original seed
        used = json.loads((run_dir.parent / "solution" / "instance.json").read_text())["seed"]
        self.assertEqual(used, 4242)
        self.assertTrue((run_dir / "data" / "rows.csv").exists())

    def test_gate_rejection_reseeds_then_succeeds(self):
        seen = []
        run_dir = self._attempt_dir()
        body = self._prepare_with(self._fake_generator(2, seen), "anytask", run_dir, 4242)
        self.assertEqual(body, "BODY")
        self.assertEqual(len(seen), 3)  # two rejected + one accepted
        self.assertEqual(seen[0], 4242)
        self.assertEqual(len(set(seen)), 3)  # each retry drew a distinct seed
        used = json.loads((run_dir.parent / "solution" / "instance.json").read_text())["seed"]
        self.assertEqual(used, seen[-1])
        self.assertNotEqual(used, 4242)

    def test_reseed_sequence_is_deterministic(self):
        seen_a, seen_b = [], []
        self._prepare_with(self._fake_generator(2, seen_a), "anytask", self._attempt_dir(), 4242)
        self._prepare_with(self._fake_generator(2, seen_b), "anytask", self._attempt_dir(), 4242)
        self.assertEqual(seen_a, seen_b)  # same start seed -> same instance, reproducible

    def test_exhausted_reseeds_raises(self):
        seen = []
        # always-failing gate: prepare gives up after _MAX_RESEEDS tries
        with self.assertRaises(RuntimeError) as ctx:
            self._prepare_with(
                self._fake_generator(10**9, seen), "anytask", self._attempt_dir(), 4242
            )
        self.assertEqual(len(seen), evals_provider._MAX_RESEEDS)
        self.assertNotIn("GateError", type(ctx.exception).__name__)  # surfaced as RuntimeError

    def test_real_mit_bad_seed_is_recovered(self):
        """Regression for the reported 13/mit failure: the seed treatment 13
        deterministically draws trips mit's scan-vs-rolling gate; prepare() must
        still yield a valid instance by reseeding. (If a future fix to mit's
        _rolling_7d makes this seed pass on the first try, this test becomes
        obsolete and should be removed.)"""
        import shutil

        import evals.feature.mit.generate as mitgen

        bad_seed = evals_provider.seed_for(
            "13_hw-cli-opt2-session-reuse-skills-opt-opus", "feature", "mit", 1
        )
        # precondition: the raw generator really does reject this seed
        staging = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, staging, True)
        with self.assertRaises(mitgen.GateError):
            mitgen.generate(bad_seed, staging)
        # prepare() recovers it: a valid instance on a reseeded value
        run_dir = self._attempt_dir()
        body = evals_provider.prepare("mit", run_dir, bad_seed)
        self.assertTrue(body)
        used = json.loads((run_dir.parent / "solution" / "instance.json").read_text())["seed"]
        self.assertNotEqual(used, bad_seed)
        self.assertTrue((run_dir / "data").is_dir())


class PlatformEnvTests(unittest.TestCase):
    """_platform_env: the KEY=VALUE handoff file a platform setup step writes
    (via $MLPAB_PLATFORM_ENV) so values it derives — e.g. the SageMaker
    execution-role ARN it just created — reach the agent's env."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        self.run_dir = Path(tempfile.mkdtemp())

    def test_missing_file_is_empty(self):
        self.assertEqual(runner._platform_env(self.run_dir), {})

    def test_parses_and_consumes_file(self):
        path = self.run_dir / "platform.env"
        path.write_text(
            "# from setup.py\n"
            "SAGEMAKER_ROLE_ARN=arn:aws:iam::123:role/mlpab-sagemaker-execution-role\n"
            "\n"
            "malformed line without equals\n"
            "SPACED = padded value \n"
        )
        env = runner._platform_env(self.run_dir)
        self.assertEqual(
            env,
            {
                "SAGEMAKER_ROLE_ARN": "arn:aws:iam::123:role/mlpab-sagemaker-execution-role",
                "SPACED": "padded value",
            },
        )
        # Consumed: deleted so it never lingers in the agent's workdir.
        self.assertFalse(path.exists())

    def test_declared_keys_win_on_merge(self):
        (self.run_dir / "platform.env").write_text("SAGEMAKER_ROLE_ARN=derived\nEXTRA=kept\n")
        keys = {"SAGEMAKER_ROLE_ARN": "from-dotenv"}
        merged = {**runner._platform_env(self.run_dir), **keys}
        self.assertEqual(merged, {"SAGEMAKER_ROLE_ARN": "from-dotenv", "EXTRA": "kept"})


class RunAuxTests(unittest.TestCase):
    """Platform setup/teardown subprocesses must not be instrumented: their REST
    traffic is plumbing, not agent work, so the api-log env vars are stripped
    before the step runs (the per-run venv's shim logs whenever MLPAB_API_LOG
    is set, and these steps inherit os.environ wholesale)."""

    def test_strips_api_log_instrumentation_env(self):
        import tempfile
        from pathlib import Path

        run_dir = Path(tempfile.mkdtemp())
        out = run_dir / "env.txt"
        env = {
            "PATH": os.environ.get("PATH", ""),
            "MLPAB_API_LOG": str(run_dir / "api_calls.jsonl"),
            "MLPAB_IFACE_SDK": "hopsworks",
            "HOPSWORKS_API_KEY": "kept",
        }
        runner._run_aux([f'env > "{out}"'], run_dir, env)
        text = out.read_text()
        self.assertNotIn("MLPAB_API_LOG", text)
        self.assertNotIn("MLPAB_IFACE_SDK", text)
        self.assertIn("HOPSWORKS_API_KEY=kept", text)
        # The caller's env dict is not mutated.
        self.assertIn("MLPAB_API_LOG", env)


class RunAuxDegradedTests(unittest.TestCase):
    """_run_aux stays best-effort (never raises), but a degraded step must be
    VISIBLE in the run console: non-zero exit, timeout, and — the wedged-cluster
    case — an HTTP 504 gateway timeout reported by a step that itself exits 0
    (setup/teardown scripts swallow their own errors by design, so the 504 only
    exists in their output). Without this, a 504-wedged platform leaves a dead
    session and an empty agent.log as the only trace (observed 2026-07-13)."""

    def setUp(self):
        import tempfile

        self.run_dir = Path(tempfile.mkdtemp())
        self.env = {"PATH": os.environ.get("PATH", "")}

    def _stderr_of(self, steps, **kwargs) -> str:
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            runner._run_aux(steps, self.run_dir, self.env, **kwargs)
        return buf.getvalue()

    def test_clean_step_prints_nothing(self):
        self.assertEqual(self._stderr_of(['echo "created project ok"']), "")

    def test_nonzero_exit_is_surfaced_with_output(self):
        err = self._stderr_of(['echo "boom reason"; exit 3'])
        self.assertIn("platform step degraded (exit 3)", err)
        self.assertIn("boom reason", err)

    def test_504_in_output_is_surfaced_despite_exit_0(self):
        step = "echo 'create gave up: HTTP code: 504, HTTP reason: Gateway Time-out'"
        err = self._stderr_of([step])
        self.assertIn("HTTP 504 gateway timeout in output", err)
        self.assertIn("Gateway Time-out", err)

    def test_timeout_is_surfaced(self):
        err = self._stderr_of(["sleep 5"], timeout=1)
        self.assertIn("timed out after 1s", err)

    def test_surfaced_output_is_redacted(self):
        env = dict(self.env, HOPSWORKS_API_KEY="supersecretvalue123")
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            runner._run_aux(
                ['echo "failed with key supersecretvalue123"; exit 1'], self.run_dir, env
            )
        err = buf.getvalue()
        self.assertIn("platform step degraded", err)
        self.assertNotIn("supersecretvalue123", err)


class DetectEnvironmentTests(unittest.TestCase):
    def test_includes_cores_and_start_method(self):
        text = runner.detect_environment()
        self.assertIn("CPU cores", text)
        self.assertIn("start method", text)

    def test_spawn_platform_warns_about_dataloader(self):
        with (
            mock.patch("platform.system", return_value="Darwin"),
            mock.patch("platform.machine", return_value="arm64"),
            mock.patch("shutil.which", return_value=None),
        ):
            text = runner.detect_environment()
        self.assertIn("'spawn'", text)
        self.assertIn("DEADLOCK", text)
        self.assertIn("num_workers", text)
        self.assertIn("MPS", text)

    def test_fork_platform_omits_deadlock_warning(self):
        with (
            mock.patch("platform.system", return_value="Linux"),
            mock.patch("platform.machine", return_value="x86_64"),
            mock.patch("shutil.which", return_value=None),
        ):
            text = runner.detect_environment()
        self.assertIn("'fork'", text)
        self.assertNotIn("DEADLOCK", text)
        self.assertIn("No CUDA GPU", text)

    def test_nvidia_present_mentions_cuda(self):
        with (
            mock.patch("platform.system", return_value="Linux"),
            mock.patch("platform.machine", return_value="x86_64"),
            mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
        ):
            text = runner.detect_environment()
        self.assertIn("CUDA", text)


class BuildPromptTests(unittest.TestCase):
    def test_fills_all_placeholders(self):
        prompt = runner._build_prompt("TASK BODY TEXT", "INTERFACE_FRAGMENT")
        self.assertIn("TASK BODY TEXT", prompt)
        self.assertIn("INTERFACE_FRAGMENT", prompt)
        # The {environment} placeholder is resolved (detected bullets present).
        self.assertIn("CPU cores", prompt)
        # No unfilled placeholders leaked through.
        self.assertNotIn("{environment}", prompt)
        self.assertNotIn("{task_body}", prompt)
        self.assertNotIn("{fragment}", prompt)

    def test_interface_under_test_section(self):
        # With an interface present (default), the under-test rule is included
        # and the HTML markers are stripped.
        on = runner._build_prompt("c", "FRAG", interface_under_test=True)
        self.assertIn("The interface is what's being measured", on)
        self.assertNotIn("UNDER_TEST_START", on)
        self.assertNotIn("UNDER_TEST_END", on)

        # For none/none the whole section is removed.
        off = runner._build_prompt("c", "FRAG", interface_under_test=False)
        self.assertNotIn("The interface is what's being measured", off)
        self.assertNotIn("UNDER_TEST", off)
        self.assertNotIn("\n\n\n", off)  # no collapsed-newline gap left behind


class RelocateVenvTests(unittest.TestCase):
    """_relocate_venv rewrites a cloned prepared venv's self-referential
    absolute path so the run venv is self-contained inside the sandbox."""

    def test_rewrites_shebangs_and_pyvenv_cfg(self):
        tmp = Path(tempfile.mkdtemp())
        old = tmp / "build" / "venv"
        new = tmp / "run" / "venv"
        (new / "bin").mkdir(parents=True)
        # A console-script shebang and an activate script carry the venv path.
        script = new / "bin" / "tool"
        script.write_text(f"#!{old}/bin/python\nprint('hi')\n")
        activate = new / "bin" / "activate"
        activate.write_text(f'VIRTUAL_ENV="{old}"\nexport VIRTUAL_ENV\n')
        (new / "pyvenv.cfg").write_text(f"home = {old}/bin\ncommand = {old}/bin/python\n")
        # A symlink (bin/python) and a real binary must be left untouched.
        (new / "bin" / "python").symlink_to("/usr/bin/python3")
        binary = new / "bin" / "realbin"
        binary.write_bytes(b"\x7fELF\x00\x00binary" + str(old).encode())

        runner._relocate_venv(old, new)

        self.assertIn(f"#!{new}/bin/python", script.read_text())
        self.assertNotIn(str(old), script.read_text())
        self.assertIn(str(new), activate.read_text())
        self.assertNotIn(str(old), activate.read_text())
        self.assertNotIn(str(old), (new / "pyvenv.cfg").read_text())
        # symlink target unchanged; binary bytes untouched.
        self.assertEqual(os.readlink(new / "bin" / "python"), "/usr/bin/python3")
        self.assertIn(str(old).encode(), binary.read_bytes())

    def test_make_venv_clones_prepared_when_present(self):
        tmp = Path(tempfile.mkdtemp())
        prepared = tmp / "prepared"
        (prepared / "bin").mkdir(parents=True)
        (prepared / "bin" / "python").write_text("#!/bin/sh\n")
        (prepared / "marker.txt").write_text("from-prepared")
        target = tmp / "run" / "venv"

        py = runner._make_venv(target, prepared=prepared)

        self.assertEqual(py, target / "bin" / "python")
        # Cloned the prepared tree rather than building a fresh base venv.
        self.assertEqual((target / "marker.txt").read_text(), "from-prepared")


class VerifyPlatformTests(unittest.TestCase):
    """_verify_platform runs a platform's `<phase>.py verify` and reports
    (ok, output); absent script → (True, '') (nothing to verify)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.run_dir = self.tmp / "run"
        self.run_dir.mkdir()
        self.configs = self.tmp / "configs"
        (self.configs / "svc").mkdir(parents=True)
        self._orig = interfaces.CONFIGS_DIR
        interfaces.CONFIGS_DIR = self.configs

    def tearDown(self):
        interfaces.CONFIGS_DIR = self._orig

    def _write(self, phase, body):
        (self.configs / "svc" / f"{phase}.py").write_text(body)

    def test_absent_script_is_ok(self):
        ok, out = runner._verify_platform("setup", "svc", self.run_dir, {})
        self.assertTrue(ok)
        self.assertEqual(out, "")

    def test_verify_pass(self):
        self._write(
            "setup", "import sys\nprint('verify' if sys.argv[1:2]==['verify'] else 'apply')\n"
        )
        ok, out = runner._verify_platform("setup", "svc", self.run_dir, {})
        self.assertTrue(ok)
        self.assertIn("verify", out)

    def test_verify_fail_surfaces_output(self):
        self._write("setup", "import sys\nprint('NO CONNECTION')\nsys.exit(1)\n")
        ok, out = runner._verify_platform("setup", "svc", self.run_dir, {})
        self.assertFalse(ok)
        self.assertIn("NO CONNECTION", out)


if __name__ == "__main__":
    unittest.main()
