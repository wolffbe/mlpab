"""Orchestrates a single (challenge, platform, interface, skills) engineer run.

Layout under the caller-supplied `runs_root`:
    <runs_root>/<task>/<challenge>/      # one engineer run; venv torn down at end
    <runs_root>/results.csv              # one detailed row per challenge

Callers place runs via `runs_root`:
    benchmark      results/benchmark/<run>/
    autoresearch   results/autoresearch/<run>/v<N>/       (per-version)

Each challenge run folder contains:
    prompt.txt            # exact task prompt handed to claude -p
    venv/                 # per-run engineer venv (deleted at teardown)
    data/                 # symlink to prepared MLE-bench data
    submission/           # where the engineer writes submission.csv
    engineer.log          # live formatted view (tee captures fd 1/2 of the run)
    commands.jsonl        # one line per tool call (cli/mcp/sdk/python/bash/skill/...)
    grading.json          # MLE-bench grader output
    endpoint_coverage.json# REST coverage vs the run's whitelist/blacklist  [when configured]
    .claude/settings.json # PreToolUse hook for claude -p
    .claude/skills/       # copied skill bundle (deleted at teardown)   [skills != none]
    .mcp.json             # MCP servers                                  [interface == mcp]

When `spec.interface_dir` is set (autoresearch's per-increment copy), the run's
platform home points at that copy via `interfaces.set_interface_home`; the stale
wheel is dropped and `interfaces.preflight` force-rebuilds the researcher's edits
before use. Login (`auth_command`) is verified per run.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from banter import (
    claude_runner,
    interfaces,
    mlebench_wrapper,
    preflight as preflight_mod,
    results,
    skills as skills_mod,
    streaming,
)


DEFAULT_MODEL = "claude-sonnet-4-6"
AUTH_MODES = ("api-key", "login")

# Testbed repo root — mle-bench data cache lives in the repo so it travels with
# the testbed and doesn't depend on the user's $HOME.
TESTBED_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = TESTBED_ROOT / "cache" / "mle-bench"
# The managed researcher venv. `banter` itself runs outside any venv; the
# researcher operates here, and each engineer run gets its own venv nested on it
# (sharing its libraries).
RESEARCHER_VENV = TESTBED_ROOT / ".venv"


@dataclass
class RunSpec:
    challenge_id: str
    platform: str                   # e.g. "hopsworks", "none"
    interface: str                  # interface: cli/mcp/sdk/none
    skills: str = "none"            # bundle name under testbed/skills/ or "none"
    docs: str = "none"              # docs bundle under platforms/<platform>/docs/ or "none"
    task: str = "no_task"           # ML task / challenge-group this challenge belongs to
    model: str = DEFAULT_MODEL
    auth: str = "api-key"
    timeout_s: int = 60 * 60
    runs_root: Path = Path("results")
    data_root: Path = DEFAULT_DATA_ROOT
    interface_version: int | None = None  # None/0 → base manifest; >0 → session version
    skills_version: int | None = None     # None → latest version
    # Session-local platform versions dir (an autoresearch session dir).
    # Required to resolve interface_version > 0.
    version_root: Path | None = None
    # Per-increment platform copy (autoresearch): engineer builds + uses THIS
    # platform home instead of the committed one. Built fresh here (copy ships
    # source, no binary). None → committed platforms/<platform>/<interface>/.
    interface_dir: Path | None = None
    # Autoresearch context (None for benchmark). `run_id` → `run` column.
    # `prev_run`/`prev_version` are continuation hints from the autoresearch
    # config, surfaced as columns so results.csv can be queried for "all runs
    # continuing from run X version v2".
    run_id: str | None = None
    version: str | None = None       # "v<N>" or just N (int as string)
    prev_run: str | None = None
    prev_version: str | None = None  # "v<N>" or just N
    # Treatment config path (experiment runs only). When set with experiment
    # metadata, the row is appended to the global results/experiments.csv instead
    # of a per-run results.csv.
    experiment_config: str | None = None
    # Run the upfront preflight (platform install/login + skill access probe).
    # Batch runners set False after one shared preflight over the union.
    preflight: bool = True


def _results_root_from(runs_root: Path) -> Path:
    """Nearest `results/` ancestor of `runs_root` (…/results/autoresearch/<run>/
    v<N>). Falls back to `runs_root` if none found."""
    runs_root = Path(runs_root)
    for p in (runs_root, *runs_root.parents):
        if p.name == "results":
            return p
    return runs_root


def _make_venv(target: Path) -> Path:
    """Create the engineer's per-run venv FULLY MATERIALIZED inside the challenge.

    The researcher venv's (testbed/.venv) site-packages are copied entirely INSIDE
    the engineer's challenge dir — no `.pth` indirection to an outside path — so
    the engineer's sandbox is truly bounded to its challenge folder.

    On APFS (macOS default) `cp -Rc` does copy-on-write clones, so the 2-3 GB
    shared site-packages costs near-zero disk until the engineer modifies a page
    (pip install of the interface). Falls back to a recursive copy where APFS
    clones aren't available (Linux, non-APFS volumes).
    """
    py = target / "bin" / "python"
    if py.exists():
        return py

    researcher_py = RESEARCHER_VENV / "bin" / "python"
    base_py = researcher_py if researcher_py.exists() else Path(sys.executable)
    subprocess.run(
        [str(base_py), "-m", "venv", "--system-site-packages", str(target)],
        check=True,
    )
    # Materialize the researcher venv's site-packages into the engineer venv: its
    # sandbox confines reads to the challenge dir, so any outside `.pth` reference
    # would resolve to a path Seatbelt denies.
    if researcher_py.exists():
        eng_sp = next((target / "lib").glob("python*/site-packages"), None)
        res_sp = next((RESEARCHER_VENV / "lib").glob("python*/site-packages"), None)
        if eng_sp and res_sp:
            _clone_tree(res_sp, eng_sp)
    return py


def _clone_tree(src: Path, dst: Path) -> None:
    """APFS-clone (or copy) every entry in `src` into existing `dst`.

    `cp -Rc` for APFS clone (near-zero cost on macOS); falls back to `cp -R` where
    clones are unsupported. `dst` already exists (created by venv), so children
    are merged in without re-creating it.
    """
    for child in src.iterdir():
        target = dst / child.name
        if target.exists():
            continue
        try:
            subprocess.run(["cp", "-Rc", str(child), str(target)], check=True)
        except subprocess.CalledProcessError:
            subprocess.run(["cp", "-R", str(child), str(target)], check=True)


def _run_aux(steps: list[str], run_dir: Path, env: dict[str, str], timeout: int = 60) -> None:
    """Best-effort platform housekeeping (serve/teardown). Failures ignored and
    output discarded — not part of the engineer's work.
    """
    for cmd in steps:
        try:
            subprocess.run(
                cmd, shell=True, cwd=str(run_dir), env=env,
                timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


TASK_PROMPT_PATH = TESTBED_ROOT / "prompts" / "engineer.md"


def _total_ram_gb() -> float | None:
    """Total physical RAM in GiB, or None if undeterminable.

    Uses POSIX sysconf (macOS and Linux); no third-party deps.
    """
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        return None


def detect_environment() -> str:
    """Detect the engineer's hardware/OS and render it as prompt bullet lines.

    Probed on the host — the same machine the per-run venv (and thus the engineer)
    runs on — so the facts apply to the engineer. Avoids importing torch (not yet
    installed at prompt-build time): accelerator availability is inferred from
    OS/arch and `nvidia-smi` presence. The `spawn`-deadlock warning is emitted only
    where the multiprocessing default is `spawn` (macOS/Windows), as the unguarded
    `num_workers>0` pattern hangs forever there.
    """
    import platform
    import shutil

    system = platform.system()                  # Darwin / Linux / Windows
    machine = platform.machine() or "?"         # arm64 / x86_64 / ...
    cpus = os.cpu_count() or 1
    ram = _total_ram_gb()
    ram_str = f", ~{ram:.0f} GB RAM" if ram else ""

    if shutil.which("nvidia-smi") is not None:
        accel = "An NVIDIA GPU is available — you may use CUDA."
    elif system == "Darwin" and machine in ("arm64", "aarch64"):
        accel = "Apple Silicon: no CUDA GPU. `torch` MPS is likely available; otherwise use CPU."
    else:
        accel = "No CUDA GPU detected — train on CPU."

    spawn = system in ("Darwin", "Windows")
    start_method = "spawn" if spawn else "fork"

    lines = [
        f"- You run on {system} ({machine}) with {cpus} CPU cores{ram_str}.",
        f"- {accel}",
        f"- Python multiprocessing uses the '{start_method}' start method.",
    ]
    if spawn:
        lines.append(
            "- Because the start method is 'spawn', a `DataLoader` with `num_workers>0` "
            "(or any multiprocessing) whose script lacks an `if __name__ == \"__main__\":` "
            "guard will DEADLOCK and hang forever — it never errors, it just stalls. "
            "Default to `num_workers=0`; only use workers if all top-level code is under "
            "that guard."
        )
    return "\n".join(lines)


# The "platform is under test" rule (engineer default), baked into engineer.md
# between these markers. Applies only when a platform is present; for none/none
# (engineer builds its own model freely) the whole section is stripped.
_UNDER_TEST_RE = re.compile(
    r"<!--UNDER_TEST_START-->\n(.*?)\n<!--UNDER_TEST_END-->", re.DOTALL
)
# Inverse: content kept ONLY when NO interface is under test (none/none, engineer
# trains its own model locally). Stripped when an interface is under test
# (everything must run remotely on the platform).
_LOCAL_ONLY_RE = re.compile(
    r"<!--LOCAL_ONLY_START-->\n(.*?)\n<!--LOCAL_ONLY_END-->", re.DOTALL
)


def _build_prompt(
    challenge_id: str,
    fragment: str,
    interface_under_test: bool = True,
) -> str:
    template = TASK_PROMPT_PATH.read_text()
    if interface_under_test:
        template = _UNDER_TEST_RE.sub(lambda m: m.group(1), template)
        template = _LOCAL_ONLY_RE.sub("", template)
    else:
        template = _UNDER_TEST_RE.sub("", template)
        template = _LOCAL_ONLY_RE.sub(lambda m: m.group(1), template)
    template = re.sub(r"\n{3,}", "\n\n", template)
    # The engineer never receives platform docs (any mode), so the prompt
    # advertises none — its only guide is the interface's own self-description.
    return template.format(
        challenge_id=challenge_id,
        fragment=fragment,
        environment=detect_environment(),
    ).strip()


def run(spec: RunSpec) -> results.Row:
    if spec.auth not in AUTH_MODES:
        raise ValueError(f"Unknown auth mode {spec.auth!r}; expected one of {AUTH_MODES}")

    started = datetime.now(timezone.utc)

    # Per-version platform copy (autoresearch): point this platform's home at the
    # copy so config/build/$INTERFACE_DIR/install all resolve there. Any wheel in
    # the copy may be stale (researcher may have edited source between
    # `prepare-version` and this run); drop it and let preflight rebuild + test.
    if spec.interface_dir is not None:
        idir = Path(spec.interface_dir)
        interfaces.set_interface_home(spec.platform, spec.interface, idir)
        # ENFORCE: an improvement version (vN, N>0) must actually change the
        # interface SOURCE vs the previous version. Prompt is frozen and
        # config.yaml excluded, so a prompt-only / no-op edit is refused here —
        # before the engineer runs — rather than silently re-measuring v(N-1).
        # Baselines / non-autoresearch runs are unaffected. Fingerprint computed
        # ONCE and reused for both the change-check and the rebuild cache (hashing
        # the tree — a full upstream checkout for Hopsworks — isn't free).
        fp = interfaces.source_fingerprint(idir)
        try:
            interfaces.assert_source_changed(idir, spec.version, fp=fp)
        except ValueError as e:
            raise preflight_mod.PreflightError(str(e))
        # Rebuild only when source changed since the last build in this copy
        # (cache keyed on the fingerprint). Repeated challenges within a version —
        # or an unchanged interface — reuse the existing wheel instead of paying a
        # full wheel build every run.
        marker = idir / ".banter-src-hash"
        wheels = list(idir.glob("*.whl"))
        if not (wheels and marker.exists() and marker.read_text().strip() == fp):
            for w in wheels:
                try:
                    w.unlink()
                except OSError:
                    pass
            st = interfaces.preflight(
                spec.platform, spec.interface, check_login=False, timeout_s=spec.timeout_s
            )
            if not st.ok:
                raise preflight_mod.PreflightError(st.message)
            marker.write_text(fp)

        # Laundering audit: flag engineer-facing interface tools the researcher
        # added/changed that run python LOCALLY (remote-only contract: tools must
        # delegate to the cluster). Diff vN against v0 so unchanged upstream
        # subprocess use isn't flagged. Surfaced as a warning + a per-version
        # `interface_audit.json` artifact — visibility, not a hard block; the
        # researcher/owner judges. Skipped for v0 (it IS the baseline).
        _vtail = str(spec.version or "").rsplit("_", 1)[-1].lstrip("v")
        ver = int(_vtail) if _vtail.isdigit() else 0
        if ver > 0:
            try:
                # Repo-backed interfaces put source under `interface/src`;
                # local / in-place ones have it directly under `interface/`.
                cur_src = (idir / "src") if (idir / "src").exists() else idir
                v0_iface = idir.parent.parent / "v0" / "interface"
                base_src = (v0_iface / "src") if (v0_iface / "src").exists() else v0_iface
                flagged = results.audit_interface_local_exec(
                    cur_src, base_src if base_src.exists() else None
                )
                if flagged:
                    (idir / "interface_audit.json").write_text(json.dumps(flagged, indent=2))
                    print("[banter] WARNING: interface tool(s) execute LOCAL python "
                          "— possible laundering of local compute as interface usage:",
                          file=sys.stderr)
                    for f in flagged:
                        print(f"  - {f['file']}: {', '.join(f['patterns'])}", file=sys.stderr)
            except Exception:
                pass  # audit is best-effort; never block a run on it

    # Fail fast: platform must be installed, login must work, and any chosen skill
    # bundle must be accessible to the engineer. Batch runners
    # (benchmark/autoresearch) preflight the union once and pass preflight=False
    # here to avoid re-probing per run.
    if spec.preflight:
        preflight_mod.check_run(
            platform=spec.platform,
            interface=spec.interface,
            interface_version=spec.interface_version,
            version_root=spec.version_root,
            skills=spec.skills,
            skills_version=spec.skills_version,
            auth=spec.auth,
            model=spec.model,
        )

    # Fail fast: validate the platform version is known before doing work (raises
    # on unknown). The resolved version/hash for the row come from
    # `interfaces.setup` below, not here.
    interfaces.variant_for(
        spec.platform, spec.interface, spec.interface_version, spec.version_root
    )
    # Fail fast on unverified skill bundles — confirm the bundle is well-formed
    # before spending time on venv/data/prep.
    if spec.skills == "none":
        skills_version, skills_hash = 0, ""
    else:
        skills_version, skills_hash, _ = skills_mod.verify_installed(
            spec.platform, spec.skills, spec.skills_version, spec.version_root
        )
    # Per-challenge output lives directly under runs_root:
    #   benchmark      results/benchmark/<run>/<task>/<challenge>/
    #   autoresearch   results/autoresearch/<run>/<increment>/<task>/<challenge>/
    # Platform/interface/skills/version are results.csv columns, not encoded in
    # the path (one platform per run, per the per-interface configs).
    run_dir = spec.runs_root / spec.task / spec.challenge_id
    # Re-runs overwrite the previous output.
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Capture the ENTIRE terminal output of this run — venv build, mle-bench data
    # prep, pip installs, engineer's streamed activity, grading — into engineer.log
    # (FD-level tee). Echo to the real terminal unless quiet or nested
    # (autoresearch, where the researcher captures this subprocess's stdout and the
    # FileTailer surfaces engineer.log live instead).
    passthrough = not streaming.nested() and not streaming.quiet()
    with streaming.tee_to(run_dir / "engineer.log", passthrough=passthrough):
        # The per-run venv clones the base .venv. Ensure the base holds no copy of
        # the interface package before cloning, so this run installs the interface
        # wheel fresh + complete (console scripts, extras).
        interfaces.ensure_base_clean(
            spec.platform, spec.interface, spec.interface_version, spec.version_root,
        )
        venv_python = _make_venv(run_dir / "venv")
        (run_dir / "submission").mkdir(exist_ok=True)

        # mle-bench data prep must succeed before spinning up Claude — without
        # data/ the engineer has nothing to score against. Don't swallow.
        mlebench_wrapper.prepare(spec.challenge_id, run_dir, spec.data_root)

        interface_setup = interfaces.setup(
            spec.platform, spec.interface, run_dir, venv_python,
            spec.interface_version, spec.version_root,
        )
        skills_setup = skills_mod.apply(
            spec.platform, spec.skills, run_dir, spec.skills_version, spec.version_root
        )
        if skills_setup.installed:
            print(
                f"[banter] skills v{skills_setup.version} ({skills_setup.hash}): "
                f"{','.join(skills_setup.installed)}",
                file=sys.stderr,
            )
        # Docs: the ENGINEER never receives platform docs, in ANY mode. Its sole
        # guide is the interface's own self-description (names, `--help`,
        # docstrings, errors) — the interface's quality is what we measure, and
        # engineer-side docs would confound that. In autoresearch the session
        # bootstrap clones the bundle into `<run>/docs/` FOR THE RESEARCHER only;
        # nothing materializes docs into the engineer's challenge dir.

        # Start fresh: stop servers a previous (possibly crashed) run left running
        # and clear leaked state (incl. platform-side artifacts the agent created —
        # see the interface `teardown:` step), then start the servers THIS run
        # needs (e.g. the MCP HTTP server claude connects to at launch).
        # `BANTER_PLATFORM_DIR` points teardown/serve steps at the COMMITTED
        # platforms/<platform>/ dir (e.g. a teardown.py cleanup script), NOT the
        # per-version interface copy — so cleanup logic is fixed infrastructure,
        # immune to interface edits.
        base_keys_env = {
            **os.environ,
            **interface_setup.keys,
            "BANTER_PLATFORM_DIR": str(interfaces.PLATFORMS_DIR / spec.platform),
        }
        _run_aux(interface_setup.teardown, run_dir, base_keys_env)
        if interface_setup.serve:
            serve_env = dict(base_keys_env)
            serve_env["PATH"] = f"{run_dir / 'venv' / 'bin'}{os.pathsep}{serve_env.get('PATH', '')}"
            serve_env["VIRTUAL_ENV"] = str(run_dir / "venv")
            _run_aux(interface_setup.serve, run_dir, serve_env)

        # Log in + verify auth for THIS challenge in its own venv. Build/test ran
        # once at session preflight; login is re-checked every run (catches expired
        # creds mid-session; each run authenticates in its own venv).
        login = interfaces.login_status(
            spec.platform, spec.interface, venv_python=venv_python, keys=interface_setup.keys,
        )
        if not login.ok:
            raise preflight_mod.PreflightError(login.message)

        prompt = _build_prompt(spec.challenge_id, interface_setup.prompt_fragment,
                               interface_under_test=interface_setup.interface != "none")
        (run_dir / "prompt.txt").write_text(prompt)

        # Autoresearch (`spec.version` set): confine the engineer's sandbox to the
        # whole version dir (`<run>/v<N>`) — runs_root IS the version dir.
        # Benchmark: None, so the engineer is confined to its own challenge dir.
        version_dir = spec.runs_root if spec.version else None
        cr = claude_runner.run(
            prompt=prompt,
            run_dir=run_dir,
            auth=spec.auth,
            model=spec.model,
            cli_binary=interface_setup.cli_binary,
            sdk_module=interface_setup.sdk_module,
            mcp_servers=interface_setup.mcp_servers,
            command_log=run_dir / "commands.jsonl",
            timeout_s=spec.timeout_s,
            extra_env=interface_setup.keys,
            version_dir=version_dir,
            allowed_domains=interface_setup.allowed_domains,
            interface=interface_setup.interface,
        )

        if cr.exit_code != 0:
            print(f"[banter] claude exit={cr.exit_code}", file=sys.stderr)

        submission = run_dir / "submission" / "submission.csv"
        try:
            grading = mlebench_wrapper.grade(spec.challenge_id, submission, spec.data_root)
        except Exception as e:
            grading = {"medal": None, "score": None, "error": str(e)}
        (run_dir / "grading.json").write_text(json.dumps(grading, indent=2))

        usage = results.parse_transcript_usage(cr.transcript_path)
        # Counting classifies by the ACTIVE interface only: a `hops`/`import
        # hopsworks` call is `cli_calls`/`sdk_calls` ONLY when that interface is
        # under test. (interface_setup markers are resolved for ALL interfaces to
        # feed the cross-interface enforcement HOOK, but counting must not relabel
        # an off-interface call as on-interface.)
        count_cli = interface_setup.cli_binary if interface_setup.interface == "cli" else None
        count_sdk = interface_setup.sdk_module if interface_setup.interface == "sdk" else None
        counts = results.aggregate_commands(
            cr.transcript_path,
            cli_binary=count_cli,
            sdk_module=count_sdk,
            run_dir=run_dir,
        )
        # Rebuild commands.jsonl from the transcript so it's always populated even
        # if the PreToolUse hook silently fails.
        results.write_commands_log(
            cr.transcript_path,
            run_dir / "commands.jsonl",
            cli_binary=count_cli,
            sdk_module=count_sdk,
            run_dir=run_dir,
        )
        # Surface the interface CLIENT's runtime logs + crashes into a per-run
        # <platform>_client.logs (cli/mcp/sdk). For mcp this rescues the stdio
        # server's stderr/connection status — a startup crash there otherwise only
        # reaches the engineer as "No such tool available", reading like an empty
        # interface. The engineer's HOME (where Claude buries the MCP logs) is the
        # version dir in autoresearch, else the challenge dir.
        if interface_setup.interface in ("cli", "mcp", "sdk"):
            boundary = (version_dir or run_dir)
            client_log = results.collect_client_logs(
                run_dir=run_dir,
                boundary=boundary,
                interface=interface_setup.interface,
                platform=spec.platform,
                mcp_servers=interface_setup.mcp_servers,
                transcript_path=cr.transcript_path,
            )
            if client_log.get("crashed"):
                print(
                    f"[banter] WARNING: {spec.platform} {interface_setup.interface} "
                    f"client crashed — see {client_log['path']} "
                    f"(markers: {', '.join(client_log['markers'])})",
                    file=sys.stderr,
                )
        # Load the experiment config ONCE (reused below for the row append). Its
        # endpoints policy drives REST-endpoint coverage scoring of the per-run API
        # log (written by the venv shim).
        exp_cfg = None
        if spec.experiment_config:
            try:
                from banter import autoresearch as ar_mod
                exp_cfg = ar_mod.load_config(Path(spec.experiment_config))
            except Exception:
                exp_cfg = None
        _endpoints = exp_cfg.endpoints if exp_cfg is not None else {"whitelist": [], "blacklist": []}
        _wl, _bl = _endpoints.get("whitelist"), _endpoints.get("blacklist")
        # Attribute whitelist coverage to the interface under test: only calls made
        # THROUGH cli/mcp/sdk count — not hand-rolled `requests`, not server-side
        # Job calls (which never reach the venv shim). A none/none baseline has no
        # interface (and no endpoints), so it stays unattributed.
        _iface = interface_setup.interface if interface_setup.interface in ("cli", "mcp", "sdk") else None
        endpoint_cov = results.endpoint_coverage(run_dir / "api_calls.jsonl", _wl, _bl, interface=_iface)
        endpoint_counts = {
            "whitelist_hits": endpoint_cov["whitelist_hits"],
            "blacklist_hits": endpoint_cov["blacklist_hits"],
        }
        # Per-run coverage breakdown the researcher reads (covered/missed target
        # endpoints) to see WHICH lifecycle steps the engineer reached — a missed
        # endpoint is a capability the interface must expose. Embeds the
        # `whitelist`/`blacklist` patterns it was scored against, so it stays
        # self-contained once the raw api_calls.jsonl is discarded below. Only when
        # configured.
        if _wl or _bl:
            (run_dir / "endpoint_coverage.json").write_text(json.dumps(endpoint_cov, indent=2))
        # mle-bench grading report → Row fields. Booleans stored 0/1 so they
        # average into rates at rollup; thresholds kept as-is (may be None).
        _b = lambda x: int(bool(x))  # noqa: E731
        grading_fields = dict(
            medal=grading.get("medal"),
            # Normalized to higher-is-better so `score: maximize` is correct even
            # across challenges with opposite native directions (raw score kept in
            # grading.json). See mlebench_wrapper.normalize_score.
            score=mlebench_wrapper.normalize_score(
                grading.get("score"), grading.get("is_lower_better")),
            valid_submission=_b(grading.get("valid_submission")),
            above_median=_b(grading.get("above_median")),
            any_medal=_b(grading.get("any_medal")),
            gold_medal=_b(grading.get("gold_medal")),
            silver_medal=_b(grading.get("silver_medal")),
            bronze_medal=_b(grading.get("bronze_medal")),
            gold_threshold=grading.get("gold_threshold"),
            silver_threshold=grading.get("silver_threshold"),
            bronze_threshold=grading.get("bronze_threshold"),
            median_threshold=grading.get("median_threshold"),
            is_lower_better=_b(grading.get("is_lower_better")),
        )
        # `version` is stored as `v<N>` (e.g. "v2"). Accept "v2" or bare "2" on
        # input — normalise to the prefixed form.
        def _to_v(v: str | None) -> str:
            if not v:
                return ""
            tail = str(v).rsplit("_", 1)[-1]
            if tail.startswith("v") and tail[1:].isdigit():
                return tail
            if tail.isdigit():
                return f"v{tail}"
            return ""
        slim_grading = {k: grading_fields.get(k) for k in ("valid_submission", "score", "medal")}
        # Map engineer's `claude -p` usage into eng_* columns. Researcher values
        # stay 0 here; autoresearch end-of-session backfills them and computes the
        # `total_*` aggregates.
        eng_wall = round(cr.wall_time_s, 2)
        eng_usage = {
            "eng_input_tokens": usage.get("input_tokens", 0),
            "eng_output_tokens": usage.get("output_tokens", 0),
            "eng_total_tokens": usage.get("total_tokens", 0),
            "eng_cost_usd": usage.get("cost_usd", 0.0),
        }
        row = results.Row(
            started_at=started.isoformat(),
            run=spec.run_id or "",
            version=_to_v(spec.version),
            platform=spec.platform,
            interface=spec.interface,
            skills=spec.skills,
            prev_run=spec.prev_run or "",
            prev_version=_to_v(spec.prev_version),
            task=spec.task,
            challenge=spec.challenge_id,
            eng_wall_time_s=eng_wall,
            **eng_usage,
            total_wall_time_s=eng_wall,
            total_tokens=eng_usage["eng_total_tokens"],
            total_cost=eng_usage["eng_cost_usd"],
            llm_calls=usage.get("llm_calls", 0),
            run_dir=str(run_dir),
            **counts,
            **endpoint_counts,
            **slim_grading,
        )

        # A DEAD run produced NO valid submission (engineer crashed / timed out
        # before writing any submission.csv). Record as a comparable score-0 row
        # with ALL numeric metrics zeroed and an `error` reason set, so the
        # researcher sees the failure (and why) without misleading partial counts
        # (the old SDK-v1 row had sdk_calls=11 / llm_calls=1 from a crash). NOTE: a
        # graceful "give up" still ships a floor submission (copy of
        # sample_submission), grading low but non-zero and NOT zeroed — that low
        # score is itself the signal.
        if not row.valid_submission:
            row.error = str(grading.get("error")
                            or "no valid submission produced (engineer crashed, "
                               "timed out, or wrote no submission.csv)")[:1000]
            # Dead-run floor. Higher-better competitions floor to 0.0 (their
            # natural worst — it drags the version's score mean down). For a
            # lower-better competition the score is sign-flipped to higher=better,
            # where 0.0 would read as the BEST possible result and silently invert
            # the ranking — so drop to None instead (excluded from the score mean;
            # the crash is still penalized via valid_submission=0 and the zeroed
            # whitelist_hits/llm_calls/… below), never rewarded.
            row.score = None if grading.get("is_lower_better") else 0.0
            for _f in ("eng_wall_time_s", "eng_input_tokens", "eng_output_tokens",
                       "eng_total_tokens", "eng_cost_usd", "total_wall_time_s",
                       "total_tokens", "total_cost", "llm_calls", "cli_calls",
                       "mcp_calls", "sdk_calls", "python_calls", "bash_calls",
                       "skill_calls", "other_tool_calls",
                       "whitelist_hits", "blacklist_hits"):
                setattr(row, _f, 0)
            print(
                f"[banter] DEAD run (no valid submission) for {run_dir} — recording "
                f"a zeroed score-0 row with error; artifacts kept on disk.",
                file=sys.stderr,
            )

        # Autoresearch writes ONE exploded row straight into the global
        # results/autoresearch/experiments.csv (no per-run results.csv); benchmark
        # keeps its per-session results.csv. `exp_cfg` loaded once above (for
        # endpoint scoring) is reused here.
        if exp_cfg is not None:
            from banter import experiments as experiments_mod
            results_root = _results_root_from(spec.runs_root)
            experiments_mod.append_run(results_root, exp_cfg, asdict(row))
        else:
            # One results.csv per session, in that session's run dir. For
            # autoresearch `runs_root` is `<run>/v<N>`, so the CSV lives one level
            # up at `<run>/results.csv`; benchmark keeps it at runs_root.
            master_csv_dir = spec.runs_root.parent if spec.version else spec.runs_root
            results.append(
                master_csv_dir / "results.csv",
                row,
                # Benchmark uses the slimmer column set; autoresearch the full FIELDS.
                fields=None if spec.version else results.BENCHMARK_FIELDS,
            )

        # Notebook regeneration happens once at end-of-autoresearch (which has the
        # goals in-process); the runner just appends rows to the CSV.

        # Teardown: run done — stop the platform's background servers, then remove
        # its standalone venv (platform package + everything the engineer
        # pip-installed) and the copied skill bundle, so nothing persists into
        # other runs. Results/artifacts (engineer.log, submission, grading,
        # commands.jsonl, endpoint_coverage.json) stay.
        _run_aux(interface_setup.teardown, run_dir, {
            **os.environ,
            **interface_setup.keys,
            "BANTER_PLATFORM_DIR": str(interfaces.PLATFORMS_DIR / spec.platform),
        })
        shutil.rmtree(run_dir / "venv", ignore_errors=True)
        shutil.rmtree(run_dir / ".claude" / "skills", ignore_errors=True)
        # Transient processing inputs, not kept per version: the raw stream-json
        # transcript (already mined for usage + commands above) and the API log
        # (already folded into endpoint_coverage.json). Drop both so each version
        # dir keeps only distilled artifacts.
        (run_dir / "transcript.jsonl").unlink(missing_ok=True)
        (run_dir / "api_calls.jsonl").unlink(missing_ok=True)
        # Claude Code's built-in sandbox is per-session config (the `sandbox` block
        # in <run_dir>/.claude/settings.json); no teardown.
    return row
