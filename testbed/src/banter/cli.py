"""banter CLI — deliberately small.

Two things only:

    banter <config.yaml>     run an autoresearch or benchmark config
    banter setup [MANIFEST]   one-time auth + credential keys (used by `make setup`)

Everything else — building interface binaries, logging in, testing they run —
happens automatically at preflight when you run a config, with no AI involved.
The researcher drives individual challenge evaluations internally via the
`run --challenge ...` form (not part of the everyday surface).

Examples:
    banter platforms/none/benchmark/config.yaml
    banter platforms/mlkit/benchmark/cli/config.yaml
    banter platforms/hopsworks/autoresearch/rq1/t01-cli-univariate.yaml   # run one treatment
    banter experiments refresh                                   # rebuild the analysis notebook
    make setup
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import click

from banter import interfaces, runner


# Testbed repo root. We anchor all per-project state here so the repo is
# self-contained regardless of where `banter` is invoked from.
TESTBED_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = TESTBED_ROOT / ".env"
KAGGLE_DIR = TESTBED_ROOT / ".kaggle"


def _load_dotenv() -> None:
    """Load `.env`, OVERRIDING the shell — else a stale export (e.g. an old
    `ANTHROPIC_API_KEY`) would win over the file the user just edited.
    `KAGGLE_CONFIG_DIR` uses setdefault so a user override is honored.
    """
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(KAGGLE_DIR))
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
    DOTENV_PATH.write_text("\n".join(f"{k}={v}" for k, v in existing.items()) + "\n")
    DOTENV_PATH.chmod(0o600)


class _ConfigGroup(click.Group):
    """A group where `banter <path>` is shorthand for `banter run <path>`."""

    def resolve_command(self, ctx, args):
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["run", *args]
        return super().resolve_command(ctx, args)


@click.group(cls=_ConfigGroup)
def main() -> None:
    """MLE-bench testbed driver for Claude Code."""
    _load_dotenv()


# ---------------------------------------------------------------------------
# banter run  (config dispatch; the --challenge form is for the researcher)
# ---------------------------------------------------------------------------


@main.command("run")
@click.argument("config", required=False, default=None, metavar="[CONFIG.yaml]")
@click.option("--challenge", default=None, help="MLE-bench competition id (single-run form).")
@click.option("--platform", "platform", default=None, help="Platform name, e.g. `hopsworks`, `none`.")
@click.option("--interface", "interface", type=click.Choice(interfaces.INTERFACES), default=None, help="Interface (cli/mcp/sdk/none).")
@click.option("--interface-version", "interface_version", type=int, default=None,
              help="Interface version. Omit/0 → base config; >0 → a session version.")
@click.option("--version-root", "version_root", type=click.Path(path_type=Path), default=None,
              help="Session dir holding versions > 0 (required when --interface-version > 0).")
@click.option("--interface-dir", "interface_dir", type=click.Path(path_type=Path), default=None,
              help="Use this interface home (a per-increment copy) instead of the committed one; "
                   "it's built + used by the engineer. Autoresearch passes <run>/<inc>/interface.")
@click.option("--run", "run_id", default=None,
              help="Autoresearch run id. Stored as the `run` column in the per-run results.csv.")
@click.option("--version", "version", default=None,
              help="Autoresearch version label (e.g. `v3` or just `3`). Stored as `version` in results.csv.")
@click.option("--prev-run", "prev_run", default=None,
              help="Autoresearch continuation hint: previous run id this one was bootstrapped from.")
@click.option("--prev-version", "prev_version", default=None,
              help="Autoresearch continuation hint: previous version (e.g. `v2`) this one was bootstrapped from.")
@click.option("--experiment-config", "experiment_config", default=None,
              help="Treatment config path (experiment runs). When it carries experiment "
                   "metadata, the row is written to the global results/experiments.csv.")
@click.option("--skills", default="none", show_default=True, help="Skill bundle name under skills/, or `none`.")
@click.option("--skills-version", "skills_version", type=int, default=None, help="Pin a skill bundle version.")
@click.option("--docs", default="none", show_default=True,
              help="Any name selects the platform's docs config at "
                   "platforms/<platform>/docs/config.yaml (its `repo:` is "
                   "git-cloned); also accepts a raw git URL, or `none`. "
                   "Materialized read-only at `<challenge>/docs/` for the engineer "
                   "(and `<run>/docs/` for the researcher).")
@click.option("--task", default="no_task", show_default=True, help="ML task / challenge group (folder in the run path).")
@click.option("--model", default=runner.DEFAULT_MODEL, show_default=True)
@click.option("--auth", type=click.Choice(runner.AUTH_MODES),
              default=lambda: os.environ.get("BANTER_AUTH", "api-key"), show_default="from .env or api-key")
@click.option("--timeout", "timeout_s", type=int, default=60 * 60, show_default=True)
@click.option("--runs-root", type=click.Path(path_type=Path), default=Path("results"), show_default=True)
@click.option("--quiet", "-q", is_flag=True, default=False,
              help="Suppress live engineer/researcher streaming to the terminal "
                   "(transcripts are still written). Same as BANTER_QUIET=1.")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Skip the overwrite prompt when this config's results dir "
                   "already exists (it will be replaced).")
def run(
    config: str | None,
    challenge: str | None,
    platform: str | None,
    interface: str | None,
    interface_version: int | None,
    version_root: Path | None,
    interface_dir: Path | None,
    run_id: str | None,
    version: str | None,
    prev_run: str | None,
    prev_version: str | None,
    experiment_config: str | None,
    skills: str,
    skills_version: int | None,
    docs: str,
    task: str,
    model: str,
    auth: str,
    timeout_s: int,
    runs_root: Path,
    quiet: bool,
    yes: bool,
) -> None:
    """Run an autoresearch/benchmark CONFIG, or a single challenge."""
    if quiet:
        # Set the env var (not just a local flag) so it reaches engineer runs
        # the researcher spawns as `banter run --challenge` subprocesses.
        os.environ["BANTER_QUIET"] = "1"
    if config is not None:
        _dispatch_config(Path(config), runs_root, assume_yes=yes)
        return

    if challenge is None:
        raise click.UsageError("Pass a CONFIG file (e.g. `banter configs/...yaml`) or use --challenge.")
    if platform is None:
        raise click.UsageError("--platform is required for the single-challenge form.")
    if interface is None:
        interface = "none" if platform == "none" else None
        if interface is None:
            raise click.UsageError("--interface is required for non-`none` platforms.")

    spec = runner.RunSpec(
        challenge_id=challenge,
        platform=platform,
        interface=interface,
        skills=skills,
        docs=docs,
        model=model,
        auth=auth,
        timeout_s=timeout_s,
        runs_root=runs_root,
        interface_version=interface_version,
        skills_version=skills_version,
        task=task,
        version_root=version_root,
        interface_dir=interface_dir,
        run_id=run_id,
        version=version,
        prev_run=prev_run,
        prev_version=prev_version,
        experiment_config=experiment_config,
    )
    row = runner.run(spec)
    click.echo(
        f"\n=== {challenge} [{row.platform}/{row.interface}, skills={row.skills}] "
        f"done in {row.eng_wall_time_s:.1f}s "
        f"({row.eng_total_tokens} tokens, ${row.eng_cost_usd:.4f}); medal={row.medal} ===\n"
        f"run dir: {row.run_dir}"
    )


def _dispatch_config(config_path: Path, runs_root: Path, assume_yes: bool = False) -> None:
    """Dispatch to autoresearch or benchmark based on config content.

    The config FILENAME stem (e.g. `rq1`, `test-skills`) names the results
    folder. `assume_yes` skips the overwrite confirmation when that folder
    already exists.
    """
    import yaml

    if not config_path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text()) or {}
    rr = (TESTBED_ROOT / "results") if runs_root == Path("results") else runs_root.resolve()
    config_name = config_path.stem

    if "goals" in data or "improve" in data:
        from banter import autoresearch as ar_mod
        cfg = ar_mod.load_config(config_path.resolve())
        ar_mod.run_autoresearch(cfg, TESTBED_ROOT, rr, config_name=config_name,
                                assume_yes=assume_yes)
    else:
        from banter import benchmark as bm_mod
        cfg = bm_mod.load_config(config_path.resolve())
        bm_mod.run_benchmark(cfg, rr, config_name=config_name, assume_yes=assume_yes)


# ---------------------------------------------------------------------------
# banter prepare-version  (copy a version's interface from the previous one)
# ---------------------------------------------------------------------------


@main.command("prepare-version")
@click.argument("interface_dir", type=click.Path(path_type=Path))
@click.option("--platform", "platform", required=True,
              help="Platform name (e.g. mlkit).")
@click.option("--interface", "interface", required=True, type=click.Choice(interfaces.INTERFACES),
              help="Interface (cli/mcp/sdk).")
@click.option("--force", is_flag=True, default=False,
              help="Overwrite an existing target directory.")
def prepare_version(
    interface_dir: Path, platform: str, interface: str, force: bool,
) -> None:
    """Auto-populate an autoresearch version's interface copy + build it.

    INTERFACE_DIR is the target — typically `<run>/v<N>/interface`. Its parent
    name (`v3`) sets N: N>0 copies the PREVIOUS version's `v<N-1>/interface`;
    N=0 (or no previous) copies the committed `platforms/<platform>/<interface>/`
    — copied in, NEVER edited in place — then BUILT so it's immediately runnable.

    The researcher calls this per new version instead of hand-rolling `cp -r` +
    install: the copy+build is deterministic, no AI touches file moves or the
    committed source roster.
    """
    target = Path(interface_dir).resolve()
    if target.exists():
        if not force:
            raise click.ClickException(
                f"{target} already exists. Pass --force to overwrite."
            )
        shutil.rmtree(target)

    v_name = target.parent.name
    if not (v_name.startswith("v") and v_name[1:].isdigit()):
        raise click.UsageError(
            f"target's parent must be v<N>, got {target.parent.name!r}."
        )
    n = int(v_name[1:])

    # v>0 → copy from the previous version if it has an interface/ dir;
    # else (or for v0) → copy from the committed base (read-only — never touched).
    src: Path | None = None
    if n > 0:
        prev = target.parent.parent / f"v{n - 1}" / "interface"
        if prev.is_dir():
            src = prev
    if src is None:
        src = TESTBED_ROOT / "platforms" / platform / interface
    if not src.is_dir():
        raise click.ClickException(f"source interface dir not found: {src}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, target)
    # Strip any wheel that came from the source so the build below is fresh.
    for w in list(target.glob("*.whl")):
        w.unlink()

    # Build IN the copy so it's runnable; for v>0 the researcher just edits
    # source and `banter run --interface-dir` force-rebuilds before each run.
    interfaces.set_interface_home(platform, interface, target)
    try:
        interfaces.build(platform, interface)
    except Exception as e:
        raise click.ClickException(f"build failed in {target}: {e}")
    built = [w.name for w in target.glob("*.whl")]
    click.echo(
        f"[prepare-version] {target}  ←  {src}"
        + (f"  (built: {', '.join(built)})" if built else "  (no wheel produced)")
    )


@main.command("test")
@click.argument("config", type=click.Path(path_type=Path))
@click.option("--no-login", is_flag=True, default=False,
              help="Skip the login/auth check (just build + run test_command).")
def test_interface(config: Path, no_login: bool) -> None:
    """Build + verify an interface ONCE, then delete the build artifacts.

    CONFIG is an interface config (platforms/<platform>/<interface>/config.yaml).
    Builds the interface, runs its auth + test_command checks, and cleans the
    build output back out so the committed folder stays source-only — a quick
    "does this interface still work?" check.
    """
    platform, interface = interfaces.platform_interface_from_config(config)
    status = interfaces.preflight(
        platform, interface, check_login=not no_login, cleanup_build=True,
    )
    if status.ok:
        click.secho(f"OK — {platform}/{interface} builds, "
                    f"{'authenticates, ' if not no_login else ''}and runs.", fg="green")
    else:
        raise click.ClickException(status.message)


@main.command("annotate-version")
@click.option("--config", "config", required=True, type=click.Path(path_type=Path),
              help="Treatment config path (the same one passed as --experiment-config).")
@click.option("--version", "version", required=True, help="Version label, e.g. `v3`.")
@click.option("--interface", "interface", default=None,
              help="Scope to one interface leaf (cli/mcp/sdk). Omit for single-interface treatments.")
@click.option("--hypothesis", default=None)
@click.option("--change", default=None)
@click.option("--verdict", type=click.Choice(["positive", "negative", "neutral"]), default=None)
@click.option("--verdict-reason", "verdict_reason", default=None)
@click.option("--keep", type=int, default=None, help="0 or 1 — did you keep the change?")
@click.option("--observations", default=None)
@click.option("--proposed-changes", "proposed_changes", default=None)
def annotate_version(
    config: Path, version: str, interface: str | None,
    hypothesis: str | None, change: str | None, verdict: str | None,
    verdict_reason: str | None, keep: int | None,
    observations: str | None, proposed_changes: str | None,
) -> None:
    """Fill the per-version annotation columns on every row of (config, version)
    in the global results/autoresearch/experiments.csv — the researcher uses this
    to record each version's hypothesis, change, verdict, and next steps. Also
    (re)writes that version's CHANGELOG.md section from the same fields + metrics,
    so the narrative is never missing."""
    from banter import experiments as experiments_mod

    config_rel = os.path.relpath(Path(config).resolve(), TESTBED_ROOT)
    updates = {
        "hypothesis": hypothesis, "change": change, "verdict": verdict,
        "verdict_reason": verdict_reason, "keep": keep,
        "observations": observations, "proposed_changes": proposed_changes,
    }
    n = experiments_mod.annotate_version(
        TESTBED_ROOT / "results", config_rel, version,
        {k: v for k, v in updates.items() if v is not None}, interface=interface,
    )
    click.echo(f"annotated {n} row(s) (config={config_rel}, version={version}"
               + (f", interface={interface}" if interface else "") + ")"
               + ("  + wrote CHANGELOG.md section" if n else ""))


@main.command("clean")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip the confirmation prompt.")
def clean(yes: bool) -> None:
    """Remove all benchmark + autoresearch runs — every session/run directory
    under results/ (which also clears any session-local interface/skill versions,
    since those live inside the autoresearch session dirs).

    The benchmark rollup CSV (results/benchmark/results.csv) is reset to a
    header-only file. Autoresearch has no global rollup — each session keeps its
    own results.csv inside its dir — so any stale global CSV is just removed."""
    from banter import results as results_mod

    results_dir = TESTBED_ROOT / "results"
    benchmark_dir = results_dir / "benchmark"
    autoresearch_dir = results_dir / "autoresearch"

    run_dirs = [
        p
        for d in (benchmark_dir, autoresearch_dir) if d.exists()
        for p in d.iterdir() if p.is_dir()
    ]
    if not run_dirs:
        click.echo("Nothing to clean.")
        return
    if not yes:
        click.echo(f"This will delete {len(run_dirs)} run/session dir(s) under {results_dir}.")
        if not click.confirm("Proceed?", default=False):
            click.echo("Aborted.")
            return

    for d in (benchmark_dir, autoresearch_dir):
        d.mkdir(parents=True, exist_ok=True)
        for p in d.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
        (d / ".gitkeep").touch()

    # Benchmark keeps a header-only rollup; autoresearch has no global rollup,
    # so drop any stale global CSV instead of recreating one.
    (benchmark_dir / "results.csv").write_text(
        ",".join(results_mod.COMBO_SUMMARY_FIELDS) + "\n"
    )
    stale_global = autoresearch_dir / "results.csv"
    if stale_global.exists():
        stale_global.unlink()

    # Drop the global experiments table (results/autoresearch/experiments.csv);
    # it's written live as treatments run, so a clean slate means removing it.
    from banter import experiments as experiments_mod
    exp_table = experiments_mod.table_path(results_dir)
    if exp_table.exists():
        exp_table.unlink()

    click.secho(f"Cleaned: removed {len(run_dirs)} run/session dir(s).", fg="green")


# ---------------------------------------------------------------------------
# banter budget-check  (graceful COMPUTE-time cap for the researcher)
# ---------------------------------------------------------------------------


@main.command("budget-check")
@click.option("--start", type=int, required=True,
              help="Session start epoch (seconds).")
@click.option("--max-seconds", type=float, required=True,
              help="Compute-time budget in seconds.")
@click.option("--ledger", type=click.Path(path_type=Path), default=None,
              help="Rate-limit-wait ledger; its total is excluded from elapsed.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Emit a machine-readable JSON line instead of prose.")
def budget_check(start: int, max_seconds: float, ledger: Path | None, as_json: bool) -> None:
    """Check the COMPUTE-time budget — wall clock MINUS rate-limit waiting.

    `compute = (now - start) - sum(ledger)`. Exit `3` when `compute >=
    max-seconds` (so the researcher can do `banter budget-check ... || finalize`),
    `0` otherwise. Rate-limit sleep time recorded in `--ledger` is excluded so
    only actual computation counts against the budget.
    """
    import time
    from banter import claude_runner

    waited = claude_runner.read_rate_limit_wait(ledger) if ledger else 0.0
    compute = max(0.0, (int(time.time()) - start) - waited)
    over = compute >= max_seconds
    if as_json:
        click.echo(json.dumps({
            "compute_s": round(compute, 1),
            "max_seconds": max_seconds,
            "rate_limit_wait_s": round(waited, 1),
            "stop": over,
        }))
    else:
        cap = "unlimited" if max_seconds == float("inf") else f"{max_seconds:.0f}s cap"
        click.echo(
            f"compute {compute:.0f}s / {cap} "
            f"(rate-limit waits excluded: {waited:.0f}s) -> {'STOP' if over else 'CONTINUE'}"
        )
    raise SystemExit(3 if over else 0)


# ---------------------------------------------------------------------------
# banter experiments  (design matrix → per-treatment configs + master table)
# ---------------------------------------------------------------------------


@main.group("experiments")
def experiments_grp() -> None:
    """Global results table + analysis notebook over the autoresearch runs."""


@experiments_grp.command("refresh")
@click.option("--no-exec", is_flag=True, default=False,
              help="Only (re)write the notebook; don't execute it.")
def experiments_refresh(no_exec: bool) -> None:
    """Regenerate AND execute the global results/autoresearch/analysis.ipynb
    against the current experiments.csv (the table is written live by runs).
    Run it any time to refresh the analysis; `--no-exec` skips execution."""
    from banter import experiments as experiments_mod

    try:
        nb = experiments_mod.build_global_notebook(TESTBED_ROOT, execute=not no_exec)
    except Exception as e:
        # The notebook file was still written (unexecuted); surface the error.
        nb = experiments_mod.table_path(TESTBED_ROOT / "results").parent / "analysis.ipynb"
        raise click.ClickException(f"notebook written but execution failed: {e}\n  {nb}")
    click.secho(f"{'Wrote' if no_exec else 'Executed'} global analysis notebook: {nb}", fg="green")


# ---------------------------------------------------------------------------
# banter setup  (auth for engineer + researcher; credential keys per interface)
# ---------------------------------------------------------------------------


@main.command("setup")
@click.argument("manifest", required=False, type=click.Path(path_type=Path))
def setup(manifest: Path | None) -> None:
    """One-time auth setup: Kaggle + Claude (engineer + researcher).

    Interface build/login/keys are preflight concerns handled when you run a
    config — preflight only points you at `banter setup <config.yaml>` if a key
    that config needs is missing. Pass a MANIFEST here to set just that
    interface's credential keys."""
    if manifest is not None:
        _setup_interface_keys(manifest.resolve())
        return
    click.echo("== banter setup ==")
    click.echo(f"Claude Code default model: {runner.DEFAULT_MODEL}")
    _setup_kaggle()
    _setup_claude_auth()
    click.echo("\nDone. Try: banter platforms/none/benchmark/config.yaml")


def _detect_platform_interface(config_path: Path) -> tuple[str, str]:
    """Infer (platform, interface) from platforms/<platform>/<interface>/config.yaml."""
    try:
        return interfaces.platform_interface_from_config(config_path)
    except ValueError as e:
        raise click.ClickException(str(e))


def _setup_kaggle() -> None:
    click.secho("\n[1/2] Kaggle credentials", bold=True)
    click.echo("  Required by mle-bench to download competition data.")
    click.echo("  Use the LEGACY API key (32 hex chars), not the new 'KGAT...' token.")
    click.echo("  Get one at https://www.kaggle.com/settings → API → Create New API Token.")
    kaggle_json = KAGGLE_DIR / "kaggle.json"
    if kaggle_json.exists():
        click.echo(f"  Existing credentials at {kaggle_json}.")
        if not click.confirm("  Overwrite?", default=False):
            return
    username = click.prompt("  Kaggle username")
    key = click.prompt("  Kaggle API key (legacy, 32 hex chars)", hide_input=True)
    if key.startswith("KGAT") or len(key) != 32:
        click.secho(
            f"  Warning: key looks unusual (length={len(key)}, prefix={key[:4]!r}). Proceeding.",
            fg="yellow",
        )
    kaggle_json.parent.mkdir(parents=True, exist_ok=True)
    kaggle_json.write_text(json.dumps({"username": username, "key": key}))
    kaggle_json.chmod(0o600)
    click.echo(f"  Wrote {kaggle_json}")


def _setup_claude_auth() -> None:
    click.secho("\n[2/2] Claude Code auth (engineer + researcher)", bold=True)
    click.echo("  Both the engineer (controlled) and researcher (controller) instances")
    click.echo("  authenticate the same way:")
    click.echo("    api-key  — claude-code calls Anthropic with ANTHROPIC_API_KEY.")
    click.echo("    login    — claude-code uses your Claude subscription via `claude auth login`.")
    click.echo("  (The model is set per run from the configs' engineer_model/researcher_model.)")
    choice = click.prompt("  Auth mode", type=click.Choice(list(runner.AUTH_MODES)), default="api-key")
    if choice == "api-key":
        api_key = click.prompt("  Anthropic API key", hide_input=True)
        _write_dotenv({"BANTER_AUTH": "api-key", "ANTHROPIC_API_KEY": api_key})
        click.echo(f"  Wrote ANTHROPIC_API_KEY to {DOTENV_PATH}")
        click.echo("  This key authenticates both the engineer and the researcher.")
    else:
        _write_dotenv({"BANTER_AUTH": "login"})
        if shutil.which("claude") is None:
            click.secho("  `claude` CLI not on PATH — install Claude Code first.", fg="yellow")
            return
        # `claude auth login` runs the sign-in flow and exits (no REPL to quit).
        if click.confirm("  Run `claude auth login` now? (logs in engineer + researcher)", default=True):
            subprocess.run(["claude", "auth", "login"], check=False)


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

    start = next((i for i, ln in enumerate(lines) if ln.rstrip() == "keys:" or ln.startswith("keys:")), None)
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
