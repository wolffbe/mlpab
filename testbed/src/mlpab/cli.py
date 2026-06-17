"""mlpab CLI — deliberately small.

    mlpab <config.yaml>      run a treatment config inline (the FTI eval grid)
    mlpab start <config>     run it in a detached tmux session instead
    mlpab status / attach / stop   manage those sessions
    mlpab setup [MANIFEST]   one-time auth + credential keys (used by `make setup`)

Everything else — building interface binaries, logging in, testing they run —
happens automatically at preflight when you run a config, with no AI involved.

Examples:
    mlpab start configs/treatments/hopsworks-cli-sdk-no-skills-opus.yaml
    mlpab test configs/platforms/hopsworks/sdk.yaml
    make setup
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import click

from mlpab import interfaces, runner

# Testbed repo root. We anchor all per-project state here so the repo is
# self-contained regardless of where `mlpab` is invoked from.
TESTBED_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = TESTBED_ROOT / ".env"


def _load_dotenv() -> None:
    """Load `.env`, OVERRIDING the shell — else a stale export (e.g. an old
    `ANTHROPIC_API_KEY`) would win over the file the user just edited.
    """
    # Hide pip's `A new release of pip is available` banner from all subprocess
    # pip calls (make install, per-run venvs, interface `install:` steps). We
    # intentionally do NOT upgrade pip in every venv.
    os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    os.environ.setdefault("PIP_NO_PYTHON_VERSION_WARNING", "1")
    if not DOTENV_PATH.exists():
        return
    for line in DOTENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


# `.env` is grouped by platform/concern so it stays readable as more platforms
# are configured: related keys cluster under a header, and groups are separated
# by a blank line (e.g. the hopsworks block is visibly apart from the gcp block).
# `_load_dotenv` ignores comments/blank lines, so the grouping is cosmetic only.
# A key is matched to the first group whose prefixes it starts with; unmatched
# keys fall into "Other" at the end. (Match on bare prefixes so AZURE also
# catches AZUREML_*.)
_DOTENV_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("Agent — Claude", ("ANTHROPIC", "CLAUDE")),
    ("Agent — Codex", ("CODEX", "OPENAI")),
    ("Agent — Mistral", ("MISTRAL",)),
    ("Hopsworks", ("HOPSWORKS",)),
    ("Databricks", ("DATABRICKS",)),
    ("AWS", ("AWS",)),
    ("Azure", ("AZURE",)),
    ("GCP", ("GCP", "GOOGLE", "CLOUDSDK")),
    ("Testbed", ("MLPAB",)),
]


def _dotenv_group_for(key: str) -> int:
    for i, (_, prefixes) in enumerate(_DOTENV_GROUPS):
        if key.startswith(prefixes):
            return i
    return len(_DOTENV_GROUPS)  # "Other" bucket, emitted last


def _write_dotenv(updates: dict[str, str]) -> None:
    existing: dict[str, str] = {}
    if DOTENV_PATH.exists():
        for line in DOTENV_PATH.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            existing[key.strip()] = value.strip()
    existing.update({k: v for k, v in updates.items() if v is not None})

    # Bucket keys by group (stable: preserves first-seen order within a group).
    buckets: dict[int, list[str]] = {}
    for key in existing:
        buckets.setdefault(_dotenv_group_for(key), []).append(key)

    names = [name for name, _ in _DOTENV_GROUPS] + ["Other"]
    sections = [
        "\n".join([f"# {names[gi]}", *(f"{k}={existing[k]}" for k in buckets[gi])])
        for gi in sorted(buckets)
    ]
    DOTENV_PATH.write_text("\n\n".join(sections) + "\n")
    DOTENV_PATH.chmod(0o600)


class _ConfigGroup(click.Group):
    """A group where `mlpab <path>` is shorthand for `mlpab run <path>`."""

    def resolve_command(self, ctx, args):
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["run", *args]
        return super().resolve_command(ctx, args)


@click.group(cls=_ConfigGroup)
def main() -> None:
    """MLPlatformAgentBench — ML-platform agent testbed (generated FTI evals) for Claude Code."""
    _load_dotenv()


# ---------------------------------------------------------------------------
# mlpab run  (config dispatch; the --task form runs one task)
# ---------------------------------------------------------------------------


@main.command("run")
@click.argument("config", required=False, default=None, metavar="[CONFIG.yaml]")
@click.option("--task", default=None, help="FTI sub-task (an evals family, e.g. `training_data`).")
@click.option(
    "--platform", "platform", default=None, help="Platform name, e.g. `hopsworks`, `none`."
)
@click.option(
    "--interface",
    "interface",
    type=click.Choice(interfaces.INTERFACES),
    default=None,
    help="Interface (cli/mcp/sdk/none).",
)
@click.option(
    "--skills", default="none", show_default=True, help="Platform skill bundle name, or `none`."
)
@click.option(
    "--category",
    default="no_task",
    show_default=True,
    help="FTI category folder (the stage the task belongs to).",
)
@click.option("--model", default=runner.DEFAULT_MODEL, show_default=True)
@click.option(
    "--auth",
    type=click.Choice(runner.AUTH_MODES),
    default=lambda: os.environ.get("MLPAB_AUTH", "api-key"),
    show_default="from .env or api-key",
)
@click.option("--timeout", "timeout_s", type=int, default=60 * 60, show_default=True)
@click.option(
    "--runs-root", type=click.Path(path_type=Path), default=Path("results"), show_default=True
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress live agent streaming to the terminal "
    "(transcripts are still written). Same as MLPAB_QUIET=1.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip the overwrite prompt when this config's results dir "
    "already exists (it will be replaced).",
)
def run(
    config: str | None,
    task: str | None,
    platform: str | None,
    interface: str | None,
    skills: str,
    category: str,
    model: str,
    auth: str,
    timeout_s: int,
    runs_root: Path,
    quiet: bool,
    yes: bool,
) -> None:
    """Run a treatment CONFIG, or a single task."""
    if quiet:
        os.environ["MLPAB_QUIET"] = "1"
    if config is not None:
        _dispatch_config(Path(config), runs_root, assume_yes=yes)
        return

    if task is None:
        raise click.UsageError("Pass a CONFIG file (e.g. `mlpab configs/...yaml`) or use --task.")
    if platform is None:
        raise click.UsageError("--platform is required for the single-task form.")
    if interface is None:
        interface = "none" if platform == "none" else None
        if interface is None:
            raise click.UsageError("--interface is required for non-`none` platforms.")

    spec = runner.RunSpec(
        task=task,
        platform=platform,
        interface=interface,
        skills=skills,
        model=model,
        auth=auth,
        timeout_s=timeout_s,
        runs_root=runs_root,
        category=category,
    )
    row = runner.run(spec)
    click.echo(
        f"\n=== {task} [{row.platform}/{row.interface}, skills={row.skills}] "
        f"done in {row.wall_time_s:.1f}s "
        f"({row.total_tokens} tokens, ${row.cost_usd:.4f}); "
        f"asserts={row.asserts_passed} passed/{row.asserts_failed} failed/"
        f"{row.asserts_skipped} skipped of {row.total_asserts} ===\n"
        f"run dir: {row.run_dir}"
    )


def _dispatch_config(config_path: Path, runs_root: Path, assume_yes: bool = False) -> None:
    """Run a treatment config. The config FILENAME STEM labels the results
    folder and the CSV `config` column — treatment configs are platform-prefixed
    (hopsworks-opus-4-8-skills.yaml, local-haiku-4-5.yaml), so stems are unique."""
    if not config_path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")
    rr = (TESTBED_ROOT / "results") if runs_root == Path("results") else runs_root.resolve()
    config_name = config_path.stem

    from mlpab import treatments as tr_mod

    cfg = tr_mod.load_config(config_path.resolve())
    tr_mod.run_treatments(cfg, rr, config_name=config_name, assume_yes=assume_yes)


# ---------------------------------------------------------------------------
# mlpab start/stop/attach/status — treatment sessions in tmux, detached from
# any terminal (or Claude). One session per config, named mlpab-<config-stem>;
# created by `start`, dies on its own when the run finishes (or via `stop`).
# Per-run detail persists in each attempt's task/agent.log regardless.
# Same-platform sessions are parallel-safe: each run mints a unique
# HOPSWORKS_PROJECT (runner.py) and setup/teardown are scoped to ONLY that
# project, so concurrent runs on one cluster never sweep each other's work.
# ---------------------------------------------------------------------------


def _tmux_session(config_path: Path) -> str:
    return f"mlpab-{config_path.stem}"


def _require_tmux() -> None:
    if shutil.which("tmux") is None:
        raise click.ClickException("tmux not installed (brew install tmux)")


def _tmux_session_exists(session: str) -> bool:
    return (
        subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0
    )


def _config_sessions(config_path: Path) -> list[str]:
    """Every live tmux session for CONFIG: the base `mlpab-<stem>` plus any
    `mlpab-<stem>-<k>` copies started while the base was busy (three terminals,
    three runs, one project each)."""
    base = _tmux_session(config_path)
    out = subprocess.run(["tmux", "ls", "-F", "#{session_name}"], capture_output=True, text=True)
    names = out.stdout.split() if out.returncode == 0 else []
    return sorted(n for n in names if n == base or n.startswith(base + "-"))


def _free_session_name(config_path: Path) -> str:
    """Pick a free session name for a new copy of CONFIG. Uses the bare
    `mlpab-<stem>` when free, else the first open `mlpab-<stem>-<k>` (k≥2) — so
    running `mlpab start <same config>` in several terminals each lands in its
    own session (and therefore its own per-run Hopsworks project) instead of
    colliding on the one name."""
    base = _tmux_session(config_path)
    if not _tmux_session_exists(base):
        return base
    k = 2
    while _tmux_session_exists(f"{base}-{k}"):
        k += 1
    return f"{base}-{k}"


@main.command("start")
@click.argument("config", type=click.Path(path_type=Path))
@click.option(
    "--no-attach",
    is_flag=True,
    default=False,
    help="Start detached without attaching (just print how to attach).",
)
def start(config: Path, no_attach: bool) -> None:
    """Run CONFIG in a tmux session (survives terminal exits) and attach to it.

    The session is created detached, so it keeps running even after you detach
    (Ctrl-b d) or your terminal closes. By default we then attach to it so you
    watch it live; pass --no-attach (or run from a non-TTY) to stay detached."""
    _require_tmux()
    if not config.exists():
        raise click.ClickException(f"Config file not found: {config}")
    # A config can run in SEVERAL terminals at once (each its own process → its
    # own per-run Hopsworks project). Don't error when one is already up; take
    # the next free session name so this `start` lands in its own session.
    session = _free_session_name(config)
    import shlex
    import sys

    mlpab_bin = Path(sys.argv[0]).resolve()
    # Explicit `run` subcommand (not the `mlpab <config>` shorthand) and shell-
    # quoted args, so ANY config path starts cleanly regardless of its name or
    # spaces/quotes — the detached session has no shell of ours to lean on.
    inner = f"{shlex.quote(str(mlpab_bin))} run {shlex.quote(str(config.resolve()))}"
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "-c",
            str(TESTBED_ROOT),
            inner,
        ],
        check=True,
    )
    click.secho(f">> started {session}", fg="green")
    click.echo(f"   stop:   mlpab stop {config}")
    # Attach so the run is visible immediately. The session stays alive on
    # detach (Ctrl-b d). Skip when asked, or when there's no terminal to attach
    # to (e.g. invoked from a script) — attaching would fail without a TTY.
    if no_attach or not sys.stdout.isatty():
        click.echo(f"   watch:  mlpab attach {config}    (detach again: Ctrl-b d)")
        return
    click.echo("   attaching… (detach with Ctrl-b d; the run keeps going)")
    subprocess.run(["tmux", "attach", "-t", session])


@main.command("stop")
@click.argument("config", type=click.Path(path_type=Path))
def stop(config: Path) -> None:
    """Kill CONFIG's tmux session(s) (they also die on their own when the run ends).

    Kills every session for the config — the base `mlpab-<stem>` and any
    `mlpab-<stem>-<k>` copies started in other terminals."""
    _require_tmux()
    sessions = _config_sessions(config)
    if not sessions:
        click.echo(f">> no session {_tmux_session(config)} (already finished?)")
    for session in sessions:
        if (
            subprocess.run(
                ["tmux", "kill-session", "-t", session], capture_output=True
            ).returncode
            == 0
        ):
            click.secho(f">> killed {session}", fg="green")
    # Reap an agent subprocess the kill may have orphaned (matched on the
    # task prompt's fixed opening line — never this or other claude sessions).
    if (
        subprocess.run(
            ["pkill", "-f", "claude -p You are completing"], capture_output=True
        ).returncode
        == 0
    ):
        click.echo(">> killed orphaned agent process")


@main.command("attach")
@click.argument("config", type=click.Path(path_type=Path))
def attach(config: Path) -> None:
    """Attach to CONFIG's tmux session (detach again with Ctrl-b d).

    Attaches to the base session, or — when only numbered copies are live — the
    first of those. Use `mlpab status` to see them all and `tmux attach -t
    <name>` to pick a specific copy."""
    _require_tmux()
    sessions = _config_sessions(config)
    if not sessions:
        raise click.ClickException(f"no session for {config} (start one with `mlpab start`)")
    if len(sessions) > 1:
        click.echo(f">> {len(sessions)} sessions live: {', '.join(sessions)} — attaching to {sessions[0]}")
    subprocess.run(["tmux", "attach", "-t", sessions[0]])


@main.command("status")
def status() -> None:
    """List running treatment sessions."""
    _require_tmux()
    out = subprocess.run(["tmux", "ls"], capture_output=True, text=True).stdout
    sessions = [ln for ln in out.splitlines() if ln.startswith("mlpab-")]
    click.echo("\n".join(sessions) if sessions else "no treatment sessions running")


@main.command("test")
@click.argument("config", type=click.Path(path_type=Path))
@click.option(
    "--no-login",
    is_flag=True,
    default=False,
    help="Skip the login/auth check (just build + run test_command).",
)
def test_interface(config: Path, no_login: bool) -> None:
    """Build + verify an interface ONCE, then delete the build artifacts.

    CONFIG is an interface config (configs/platforms/<platform>/<interface>.yaml).
    Builds the interface, runs its auth + test_command checks, and cleans the
    build output back out so the committed folder stays source-only — a quick
    "does this interface still work?" check.
    """
    platform, interface = interfaces.platform_interface_from_config(config)
    status = interfaces.preflight(
        platform,
        interface,
        check_login=not no_login,
        cleanup_build=True,
    )
    if status.ok:
        click.secho(
            f"OK — {platform}/{interface} builds, "
            f"{'authenticates, ' if not no_login else ''}and runs.",
            fg="green",
        )
    else:
        raise click.ClickException(status.message)


@main.command("clean")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip the confirmation prompt.")
def clean(yes: bool) -> None:
    """Remove all runs — every run directory under results/. The global
    results CSV is reset to a header-only file."""
    from mlpab import results as results_mod

    results_dir = TESTBED_ROOT / "results"

    # Every directory under results/ is a run dir (the per-config folders).
    # The global files (results.csv, results.ipynb, .gitkeep) are not run dirs.
    run_dirs = [p for p in results_dir.iterdir() if p.is_dir()] if results_dir.exists() else []
    if not run_dirs:
        click.echo("Nothing to clean.")
        return
    if not yes:
        click.echo(f"This will delete {len(run_dirs)} run dir(s) under {results_dir}.")
        if not click.confirm("Proceed?", default=False):
            click.echo("Aborted.")
            return

    results_dir.mkdir(parents=True, exist_ok=True)
    for p in run_dirs:
        shutil.rmtree(p)
    (results_dir / ".gitkeep").touch()

    (results_dir / "results.csv").write_text(",".join(results_mod.RESULTS_SUMMARY_FIELDS) + "\n")
    stale_nb = results_dir / "results.ipynb"
    if stale_nb.exists():
        stale_nb.unlink()

    click.secho(f"Cleaned: removed {len(run_dirs)} run dir(s).", fg="green")


# ---------------------------------------------------------------------------
# mlpab setup  (auth for the agent; credential keys per interface)
# ---------------------------------------------------------------------------


@main.command("check")
@click.argument("config", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--no-llm",
    is_flag=True,
    default=False,
    help="Skip the live LLM probe (just check platforms + engine readiness).",
)
@click.option(
    "--no-platform",
    is_flag=True,
    default=False,
    help="Skip the live platform probe (just check the LLMs).",
)
@click.option(
    "--llm-timeout",
    type=int,
    default=60,
    show_default=True,
    help="Seconds to wait for each model's live probe.",
)
def check(config: Path, no_llm: bool, no_platform: bool, llm_timeout: int) -> None:
    """Readiness check for a treatment CONFIG: can it actually run? Verifies LIVE
    that each model's LLM is reachable + responsive (a one-word probe call) and
    that each platform is reachable + responsive (build + live auth/test). Exits
    non-zero if anything isn't ready."""
    from mlpab import treatments

    _load_dotenv()  # creds live in .env
    cfg = treatments.load_config(config)
    failures: list[str] = []

    # 1) LLMs — live responsiveness (subsumes the static CLI + auth check).
    if not no_llm:
        click.echo("\n[mlpab] LLM reachability (live probe):")
        _, llm_report = treatments.check_llm_live(cfg, timeout_s=llm_timeout)
        for model, ok, detail in llm_report:
            click.secho(f"  {'✓' if ok else '✗'} {model:20} {detail}", fg="green" if ok else "red")
            if not ok:
                failures.append(f"LLM {model}: {detail}")
    else:
        ready, report = treatments.check_readiness(cfg)
        treatments.print_readiness(report)
        if not ready:
            failures.extend(f"engine {m}: {d}" for m, ok, d in report if not ok)

    # 2) Platforms — live reachability (build + auth_command + test_command).
    if not no_platform:
        click.echo("\n[mlpab] platform reachability (live probe):")
        seen: set = set()
        for e in cfg.runs:
            if e.platform == "none" or (e.platform, e.interface) in seen:
                continue
            seen.add((e.platform, e.interface))
            status = interfaces.preflight(
                e.platform,
                e.interface,
                check_login=True,
                cleanup_build=False,
            )
            label = f"{e.platform}/{e.interface}"
            if status.ok:
                click.secho(f"  ✓ {label:24} reachable + responsive", fg="green")
            else:
                click.secho(f"  ✗ {label:24} {status.message}", fg="red")
                failures.append(f"platform {label}: {status.reason or status.message}")

    if not failures:
        click.secho("\nREADY ✓ — config can run.", fg="green")
    else:
        click.secho(f"\nNOT READY ✗ — {len(failures)} problem(s):", fg="red")
        for f in failures:
            click.echo(f"  • {f}")
        click.secho("Fix with `make setup` / the platform's bootstrap, then re-check.", fg="red")
        raise SystemExit(1)


@main.command("setup")
@click.argument("manifest", required=False, type=click.Path(path_type=Path))
def setup(manifest: Path | None) -> None:
    """One-time auth setup: Claude (the agent).

    Interface build/login/keys are preflight concerns handled when you run a
    config — preflight only points you at `mlpab setup <config.yaml>` if a key
    that config needs is missing. Pass a MANIFEST here to set just that
    interface's credential keys."""
    if manifest is not None:
        _setup_interface_keys(manifest.resolve())
        return
    click.echo("== mlpab setup ==")
    # 1) Agent engines (which coding agents to authenticate).
    agents = _multi_choose(
        "agent engine(s)",
        {
            "claude": "Claude Code — claude-* models",
            "codex": "OpenAI Codex — gpt-* models",
            "mistral": "Mistral Vibe — mistral-* models",
        },
    )
    for a in agents:
        _setup_agent(a)
    # 2) Platforms (which clouds to set up; azure/gcp are assisted bootstraps).
    clouds = _multi_choose(
        "platform(s)",
        {
            "hopsworks": "Hopsworks — ASSISTED (bootstrap.sh: HOPSWORKS_HOST + API_KEY)",
            "databricks": "Databricks — ASSISTED (bootstrap.sh: DATABRICKS_HOST + TOKEN)",
            "aws": "Amazon SageMaker — ASSISTED (bootstrap.sh: AWS creds + IAM policy + role)",
            "azure": "Azure ML — ASSISTED (bootstrap.sh: CLI+login+resources)",
            "gcp": "GCP / Gemini Enterprise — ASSISTED (bootstrap.sh)",
        },
    )
    for c in clouds:
        _setup_cloud(c)
    click.echo("\nDone. Verify a config is runnable with:  mlpab check <config.yaml>")


def _multi_choose(label: str, options: dict[str, str]) -> list[str]:
    """Prompt for a comma-separated subset of `options` (or 'all'/'none')."""
    click.secho(f"\nSet up which {label}?", bold=True)
    for k, desc in options.items():
        click.echo(f"    {k:11} {desc}")
    raw = click.prompt(f"  comma-separated ({', '.join(options)}), 'all', or 'none'", default="all")
    raw = raw.strip().lower()
    if raw in ("", "none"):
        return []
    if raw == "all":
        return list(options)
    picked = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in picked if x not in options]
    if bad:
        click.secho(f"  ignoring unknown: {', '.join(bad)}", fg="yellow")
    return [x for x in picked if x in options]


def _setup_agent(agent: str) -> None:
    if agent == "claude":
        _setup_claude_auth()
        return
    if agent == "codex":
        click.secho("\nCodex (gpt-* models)", bold=True)
        key = click.prompt(
            "  OPENAI_API_KEY (blank to use `codex login`)",
            default="",
            hide_input=True,
            show_default=False,
        )
        if key:
            _write_dotenv({"OPENAI_API_KEY": key})
            click.secho(f"  wrote OPENAI_API_KEY to {DOTENV_PATH}", fg="green")
        elif shutil.which("codex") and click.confirm("  Run `codex login` now?", default=True):
            subprocess.run(["codex", "login"], check=False)
        return
    if agent == "mistral":
        click.secho("\nMistral Vibe (mistral-* models)", bold=True)
        key = click.prompt("  MISTRAL_API_KEY", hide_input=True, show_default=False)
        if key:
            _write_dotenv({"MISTRAL_API_KEY": key})
            click.secho(f"  wrote MISTRAL_API_KEY to {DOTENV_PATH}", fg="green")


def _setup_cloud(platform: str) -> None:
    # Assisted path: any platform that ships a bootstrap.sh runs it (it collects
    # creds and provisions whatever the platform needs — IAM policy, resources).
    script = interfaces.CONFIGS_DIR / platform / "bootstrap.sh"
    if script.exists():
        env = dict(os.environ)
        if platform == "gcp":
            proj = click.prompt(
                "  GCP project id (GCP_PROJECT)", default=os.environ.get("GCP_PROJECT", "")
            )
            if proj:
                env["PROJECT"] = proj
        click.secho(f"\nAssisting {platform} setup via {script.name} …", bold=True)
        subprocess.run(["bash", str(script)], env=env, check=False)
        return
    # Fallback (no bootstrap.sh): prompt the platform's declared keys into .env
    keys = interfaces.keys_for(platform, "sdk")
    if not keys:
        click.secho(f"  {platform}: no keys declared", fg="yellow")
        return
    click.secho(f"\n{platform} credentials → {DOTENV_PATH}", bold=True)
    updated = {}
    for k, cur in keys.items():
        val = click.prompt(
            f"    {k} {'(set)' if cur else ''}",
            default=cur or "",
            hide_input=True,
            show_default=False,
        )
        if val:
            updated[k] = val
    if updated:
        _write_dotenv(updated)
        click.secho(f"  wrote {len(updated)} key(s)", fg="green")


def _detect_platform_interface(config_path: Path) -> tuple[str, str]:
    """Infer (platform, interface) from configs/platforms/<platform>/<interface>.yaml."""
    try:
        return interfaces.platform_interface_from_config(config_path)
    except ValueError as e:
        raise click.ClickException(str(e))


def _setup_claude_auth() -> None:
    click.secho("\n[1/1] Claude Code auth (the agent)", bold=True)
    click.echo("  Auth modes:")
    click.echo("    api-key  — claude-code calls Anthropic with ANTHROPIC_API_KEY.")
    click.echo("    login    — claude-code uses your Claude subscription via a long-lived")
    click.echo("               `claude setup-token` token (survives overnight headless runs).")
    click.echo("  This step provisions the credential into .env. The mode itself is")
    click.echo("  still set per run by the config's `auth:` field (e.g. `auth: login`).")
    choice = click.prompt(
        "  Auth mode", type=click.Choice(list(runner.AUTH_MODES)), default="api-key"
    )
    if choice == "api-key":
        api_key = click.prompt("  Anthropic API key", hide_input=True)
        _write_dotenv({"ANTHROPIC_API_KEY": api_key})
        click.echo(f"  Wrote ANTHROPIC_API_KEY to {DOTENV_PATH}")
        click.echo("  Configs that should use it need `auth: api-key` (or omit `auth:`).")
    else:
        if shutil.which("claude") is None:
            click.secho("  `claude` CLI not on PATH — install Claude Code first.", fg="yellow")
            return
        # `claude setup-token` mints a LONG-LIVED OAuth token (~1yr) for headless
        # runs. We pin it in .env as CLAUDE_CODE_OAUTH_TOKEN — resolve_oauth_token()
        # checks env first, so it wins over the short-lived Keychain token, which
        # only refreshes during interactive sessions and otherwise expires
        # mid-run overnight (silently zeroing every subsequent task).
        if click.confirm("  Run `claude setup-token` to mint a long-lived token now?", default=True):
            token = _mint_setup_token()
            if token:
                _write_dotenv({"CLAUDE_CODE_OAUTH_TOKEN": token})
                click.secho(f"  Wrote CLAUDE_CODE_OAUTH_TOKEN to {DOTENV_PATH}", fg="green")
            else:
                click.secho(
                    "  Could not capture a token from `claude setup-token`. "
                    "Run it manually and paste the sk-ant-oat01-… value into .env "
                    "as CLAUDE_CODE_OAUTH_TOKEN.",
                    fg="yellow",
                )
        click.echo(
            "  Configs that should use it need `auth: login` (already the default in shipped configs)."
        )


# Long-lived OAuth tokens from `claude setup-token` carry this prefix.
_OAUTH_TOKEN_RE = re.compile(r"sk-ant-oat01-[A-Za-z0-9_-]+")


def _mint_setup_token() -> str | None:
    """Run `claude setup-token` and return the minted long-lived token, or None.

    The sign-in flow (browser + prompts) streams to stderr/the terminal; the
    token is printed to stdout, which we capture and scan. If capture fails
    (e.g. a future CLI prints elsewhere), we fall back to prompting for a paste.
    """
    proc = subprocess.run(
        ["claude", "setup-token"],
        check=False,
        stdout=subprocess.PIPE,
        text=True,
    )
    match = _OAUTH_TOKEN_RE.search(proc.stdout or "")
    if match:
        return match.group(0)
    pasted = click.prompt(
        "  Paste the sk-ant-oat01-… token from above (blank to skip)",
        default="",
        hide_input=True,
        show_default=False,
    ).strip()
    match = _OAUTH_TOKEN_RE.search(pasted)
    return match.group(0) if match else (pasted or None)


def _setup_interface_keys(manifest_path: Path) -> None:
    if not manifest_path.exists():
        raise click.ClickException(f"Manifest not found: {manifest_path}")
    platform, interface = _detect_platform_interface(manifest_path)
    keys = interfaces.keys_for(platform, interface)
    if not keys:
        click.echo(f"  {platform}/{interface}: no `keys:` declared in the config.")
        return
    click.secho(f"  Credentials for {platform}/{interface}:", bold=True)
    updated: dict[str, str] = {}
    for k, cur in keys.items():
        shown = "(set)" if cur else "(empty)"
        val = click.prompt(f"    {k} {shown}", default=cur, hide_input=True, show_default=False)
        updated[k] = val
    _write_manifest_keys(manifest_path, updated)
    click.secho(f"  Wrote keys into {manifest_path}", fg="green")


def _write_manifest_keys(manifest_path: Path, keys: dict[str, str]) -> None:
    """Splice a `keys:` block into the manifest, preserving its comments.

    Replaces an existing top-level `keys:` block (its line plus the indented
    lines under it) or appends one at the end. Text splice (not a YAML rewrite)
    so the manifest's comments survive.
    """
    text = manifest_path.read_text()
    lines = text.splitlines()
    rendered = ["keys:"] + [f'  {k}: "{v}"' for k, v in keys.items()]

    start = next(
        (i for i, ln in enumerate(lines) if ln.rstrip() == "keys:" or ln.startswith("keys:")), None
    )
    if start is None:
        new_lines = lines + ([""] if (lines and lines[-1].strip()) else []) + rendered
    else:
        end = start + 1
        while end < len(lines) and (not lines[end].strip() or lines[end].startswith((" ", "\t"))):
            end += 1
        new_lines = lines[:start] + rendered + lines[end:]
    manifest_path.write_text("\n".join(new_lines) + "\n")
    manifest_path.chmod(0o600)


if __name__ == "__main__":
    main()
