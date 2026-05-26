"""banter CLI.

Examples:
    banter setup                                                         # one-time creds setup
    banter run --challenge titanic --interface none
    banter run --challenge titanic --interface hopsworks --mode cli
    banter run --challenge titanic --interface hopsworks --mode cli --skills hopsworks-essentials
    banter run configs/autoresearch_rq1_hopsworks.yaml                   # autoresearch dispatch
    banter run configs/benchmark_smoke_test.yaml                         # benchmark dispatch
    banter install configs/interfaces/hopsworks/cli.yaml                 # one-time install
    banter uninstall configs/interfaces/hopsworks/cli.yaml               # remove binary
    banter interfaces                                                     # list configured interfaces
    banter skills                                                         # list available skill bundles
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import click

from banter import interfaces, runner, skills


# Testbed repo root. We anchor all per-project state here so the repo is
# self-contained regardless of where `banter` is invoked from.
TESTBED_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = TESTBED_ROOT / ".env"
KAGGLE_DIR = TESTBED_ROOT / ".kaggle"


def _load_dotenv() -> None:
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(KAGGLE_DIR))
    if not DOTENV_PATH.exists():
        return
    for line in DOTENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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


@click.group()
def main() -> None:
    """MLE-bench testbed driver for Claude Code."""
    _load_dotenv()


# ---------------------------------------------------------------------------
# banter run
# ---------------------------------------------------------------------------


@main.command("run")
@click.argument("config", required=False, default=None, metavar="[CONFIG.yaml]")
@click.option("--challenge", default=None, help="MLE-bench competition id, e.g. `titanic`.")
@click.option(
    "--interface",
    "interface_name",
    default=None,
    help="Interface name (e.g. `hopsworks`, `none`).",
)
@click.option(
    "--mode",
    type=click.Choice(interfaces.TYPES),
    default=None,
    help="Interface type: cli, mcp, sdk, or none.",
)
@click.option(
    "--interface-version",
    "interface_version",
    type=int,
    default=None,
    help="Pin to a specific interface version (integer). Omit → latest in manifest.",
)
@click.option(
    "--skills",
    default="none",
    show_default=True,
    help="Skill bundle name under configs/skills/, or `none`.",
)
@click.option(
    "--skills-version",
    "skills_version",
    type=int,
    default=None,
    help="Pin to a specific skill bundle version (integer). Omit → latest.",
)
@click.option("--model", default=runner.DEFAULT_MODEL, show_default=True)
@click.option(
    "--auth",
    type=click.Choice(runner.AUTH_MODES),
    default=lambda: os.environ.get("BANTER_AUTH", "api-key"),
    show_default="from .env or api-key",
)
@click.option("--timeout", "timeout_s", type=int, default=60 * 60, show_default=True)
@click.option(
    "--runs-root",
    type=click.Path(path_type=Path),
    default=Path("results"),
    show_default=True,
)
def run(
    config: str | None,
    challenge: str | None,
    interface_name: str | None,
    mode: str | None,
    interface_version: int | None,
    skills: str,
    skills_version: int | None,
    model: str,
    auth: str,
    timeout_s: int,
    runs_root: Path,
) -> None:
    """Run one challenge or dispatch an autoresearch/benchmark config file."""
    if config is not None:
        _dispatch_config(Path(config), runs_root)
        return

    if challenge is None:
        raise click.UsageError("--challenge is required when not passing a config file.")
    if interface_name is None:
        raise click.UsageError("--interface is required when not passing a config file.")
    if mode is None:
        if interface_name == "none":
            mode = "none"
        else:
            raise click.UsageError("--mode is required for non-`none` interfaces.")

    spec = runner.RunSpec(
        challenge_id=challenge,
        interface=interface_name,
        mode=mode,
        skills=skills,
        model=model,
        auth=auth,
        timeout_s=timeout_s,
        runs_root=runs_root,
        interface_version=interface_version,
        skills_version=skills_version,
    )
    row = runner.run(spec)
    click.echo(
        f"\n=== {challenge} [{row.interface}/{row.mode}, skills={row.skills}] "
        f"done in {row.wall_time_s:.1f}s "
        f"({row.total_tokens} tokens, ${row.cost_usd:.4f}); medal={row.medal} ===\n"
        f"run dir: {row.run_dir}"
    )


def _dispatch_config(config_path: Path, runs_root: Path) -> None:
    """Dispatch to autoresearch or benchmark based on config content."""
    import yaml

    if not config_path.exists():
        raise click.ClickException(f"Config file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text()) or {}
    rr = (TESTBED_ROOT / "results") if runs_root == Path("results") else runs_root.resolve()

    if "goals" in data or "improve" in data:
        from banter import autoresearch as ar_mod
        cfg = ar_mod.load_config(config_path.resolve())
        ar_mod.run_autoresearch(cfg, TESTBED_ROOT, rr)
    else:
        from banter import benchmark as bm_mod
        cfg = bm_mod.load_config(config_path.resolve())
        bm_mod.run_benchmark(cfg, rr)


# ---------------------------------------------------------------------------
# banter install / uninstall
# ---------------------------------------------------------------------------


@main.command("install")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
def install_cmd(config_path: Path) -> None:
    """Install an interface from a manifest YAML.

    Example: banter install configs/interfaces/hopsworks/cli.yaml
    """
    _install_from_manifest(config_path.resolve())


@main.command("uninstall")
@click.argument("config_path", type=click.Path(path_type=Path))
def uninstall_cmd(config_path: Path) -> None:
    """Remove a previously installed interface (binary + generated runtime config).

    Example: banter uninstall configs/interfaces/hopsworks/cli.yaml
    """
    _uninstall_from_manifest(config_path.resolve())


def _detect_name_type(config_path: Path) -> tuple[str, str]:
    """Infer (name, type) from a manifest path like configs/interfaces/<name>/<type>.yaml."""
    parts = config_path.parts
    try:
        idx = list(parts).index("interfaces")
        name = parts[idx + 1]
        type_ = Path(parts[idx + 2]).stem  # strip ".yaml"
    except (ValueError, IndexError):
        raise click.ClickException(
            f"Cannot infer interface name/type from {config_path}. "
            "Expected path: configs/interfaces/<name>/<type>.yaml"
        )
    if type_ not in interfaces.TYPES:
        raise click.ClickException(
            f"Unknown interface type {type_!r}; expected one of {interfaces.TYPES}"
        )
    return name, type_


def _install_from_manifest(config_path: Path) -> None:
    import yaml

    cfg = yaml.safe_load(config_path.read_text()) or {}
    name, type_ = _detect_name_type(config_path)

    repo = cfg.get("repo")
    ref = cfg.get("ref", "main")
    install_steps = cfg.get("install") or []
    auth_command = cfg.get("auth_command")
    binary = cfg.get("binary")

    venv_bin = TESTBED_ROOT / ".venv" / "bin"
    # Binary artifacts go here (interfaces/<name>/<type>/0/).
    bins_dir = TESTBED_ROOT / "interfaces" / name / type_ / "0"
    bins_dir.mkdir(parents=True, exist_ok=True)
    # Source / build cache (separate from binary output).
    src_dir = TESTBED_ROOT / "cache" / "interfaces" / name / type_ / "src"

    # 1 — clone / update the source repo (if any)
    if repo:
        if (src_dir / ".git").exists():
            click.echo(f"[install] Updating {src_dir}")
            subprocess.run(["git", "-C", str(src_dir), "pull"], check=True)
        else:
            click.echo(f"[install] Cloning {repo} → {src_dir}")
            src_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", ref, repo, str(src_dir)],
                check=True,
            )

    # 2 — run one-time build/install steps
    if install_steps:
        pip_cache = TESTBED_ROOT / "cache" / "pip"
        pip_cache.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PATH"] = f"{bins_dir}{os.pathsep}{venv_bin}{os.pathsep}{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(TESTBED_ROOT / ".venv")
        env["INTERFACE_DIR"] = str(bins_dir)
        env["PIP_CACHE_DIR"] = str(pip_cache)
        cwd = str(src_dir) if src_dir.exists() else str(bins_dir)
        for step in install_steps:
            click.echo(f"[install] $ {step}")
            subprocess.run(step, shell=True, env=env, cwd=cwd, check=True)

    # 3 — run interactive auth; the CLI saves credentials to its own config.
    # We add bins_dir AND venv_bin to PATH so a freshly-built binary is callable.
    if auth_command:
        env = os.environ.copy()
        env["PATH"] = f"{bins_dir}{os.pathsep}{venv_bin}{os.pathsep}{env.get('PATH', '')}"
        env["INTERFACE_DIR"] = str(bins_dir)
        click.echo(f"\n[install] Running: {auth_command}")
        click.echo("[install] Follow the prompts to log in.")
        subprocess.run(auth_command, shell=True, env=env)

    # 4 — report result
    if binary:
        binary_path = bins_dir / binary
        if binary_path.exists():
            click.secho(f"\n[install] {name}/{type_} ready. Binary: {binary_path}", fg="green")
        else:
            click.secho(
                f"\n[install] Warning: '{binary}' not found at {binary_path} after install.",
                fg="yellow",
            )
    else:
        click.secho(f"\n[install] {name}/{type_} installed.", fg="green")


def _uninstall_from_manifest(config_path: Path) -> None:
    name, type_ = _detect_name_type(config_path)

    bins_dir = TESTBED_ROOT / "interfaces" / name / type_
    src_dir = TESTBED_ROOT / "cache" / "interfaces" / name / type_

    removed = []
    if bins_dir.exists():
        if click.confirm(f"Remove binary dir {bins_dir}?", default=True):
            shutil.rmtree(bins_dir)
            removed.append(str(bins_dir))
    if src_dir.exists():
        if click.confirm(f"Remove source cache {src_dir}?", default=False):
            shutil.rmtree(src_dir)
            removed.append(str(src_dir))

    if removed:
        click.secho(f"[uninstall] Removed: {', '.join(removed)}", fg="green")
    else:
        click.echo(f"[uninstall] Nothing removed for {name}/{type_}.")


# ---------------------------------------------------------------------------
# banter reset
# ---------------------------------------------------------------------------


@main.command("reset")
@click.argument("config_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--delete-sessions/--keep-sessions",
    default=True,
    help="Delete all results/autoresearch/* session directories.",
)
def reset_cmd(config_path: Path, delete_sessions: bool) -> None:
    """Reset autoresearch state for a config: drop improved versions and sessions.

    Example: banter reset configs/autoresearch_rq1_hopsworks.yaml

    Removes versions > 0 from the starting interface's manifest, deletes the
    matching interfaces/<name>/<type>/<v>/ binary folders, and (by default)
    deletes all results/autoresearch/*. Base version 0 is preserved.
    """
    _reset_from_config(config_path.resolve(), delete_sessions=delete_sessions)


def _reset_from_config(config_path: Path, delete_sessions: bool) -> None:
    import yaml as _yaml

    data = _yaml.safe_load(config_path.read_text()) or {}
    if "goals" not in data and "improve" not in data:
        raise click.ClickException(
            f"{config_path} is not an autoresearch config (no `goals` or `improve` key)."
        )

    # Accept both `starting_interfaces` (plural list) and the legacy
    # `starting_interface` (single dict).
    raw = data.get("starting_interfaces")
    if raw is None and "starting_interface" in data:
        single = data["starting_interface"]
        raw = [single] if isinstance(single, dict) else []
    raw = raw or []
    if not raw:
        click.echo("[reset] No starting interfaces to reset.")
    else:
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            type_ = entry.get("mode")
            if not name or not type_ or (name == "none" and type_ == "none"):
                click.echo(f"[reset] Skipping {entry} (none/none or incomplete).")
                continue
            _reset_interface_versions(name, type_)

    if delete_sessions:
        sessions_dir = TESTBED_ROOT / "results" / "autoresearch"
        if sessions_dir.exists():
            count = sum(1 for _ in sessions_dir.iterdir() if _.is_dir())
            if count and click.confirm(
                f"Remove all {count} autoresearch session(s) under {sessions_dir}?",
                default=True,
            ):
                shutil.rmtree(sessions_dir)
                click.secho(f"[reset] Removed {sessions_dir}", fg="green")
            else:
                click.echo("[reset] Kept session directories.")
        else:
            click.echo(f"[reset] No sessions to remove ({sessions_dir} absent).")


def _reset_interface_versions(name: str, type_: str) -> None:
    """Drop versions > 0 from the manifest YAML and remove their binary folders."""
    import yaml as _yaml

    mpath = TESTBED_ROOT / "configs" / "interfaces" / name / f"{type_}.yaml"
    if not mpath.exists():
        click.echo(f"[reset] No manifest at {mpath}; skipping interface reset.")
        return

    manifest = _yaml.safe_load(mpath.read_text()) or {}
    versions = manifest.get("versions") or {}
    to_drop: list[int] = []
    if isinstance(versions, dict):
        kept = {}
        for k, v in versions.items():
            try:
                ki = int(k)
            except (TypeError, ValueError):
                continue
            if ki == 0:
                kept[ki] = v
            else:
                to_drop.append(ki)
        manifest["versions"] = kept
    elif isinstance(versions, list):
        kept_list = []
        for entry in versions:
            if not isinstance(entry, dict):
                continue
            try:
                vnum = int(entry.get("version", 0))
            except (TypeError, ValueError):
                continue
            if vnum == 0:
                kept_list.append(entry)
            else:
                to_drop.append(vnum)
        manifest["versions"] = kept_list

    if to_drop:
        mpath.write_text(_yaml.dump(manifest, default_flow_style=False, sort_keys=False))
        click.secho(
            f"[reset] {mpath}: dropped versions {sorted(to_drop)}, kept 0.",
            fg="green",
        )
    else:
        click.echo(f"[reset] {mpath}: no improved versions to drop.")

    bins_root = TESTBED_ROOT / "interfaces" / name / type_
    if bins_root.exists():
        for sub in bins_root.iterdir():
            if not sub.is_dir() or not sub.name.isdigit():
                continue
            if int(sub.name) > 0:
                shutil.rmtree(sub)
                click.echo(f"[reset] Removed {sub}")


# ---------------------------------------------------------------------------
# banter autoresearch / benchmark (explicit commands)
# ---------------------------------------------------------------------------


@main.command("autoresearch")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="YAML autoresearch config.",
)
@click.option(
    "--runs-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override results directory. Defaults to testbed/results/.",
)
def autoresearch_cmd(config_path: Path, runs_root: Path | None) -> None:
    """Launch a manager Claude that iteratively improves interfaces and skills."""
    from banter import autoresearch as ar_mod

    config = ar_mod.load_config(config_path.resolve())
    rr = (runs_root or TESTBED_ROOT / "results").resolve()
    ar_mod.run_autoresearch(config, TESTBED_ROOT, rr)


@main.command("benchmark")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="YAML benchmark config.",
)
@click.option(
    "--runs-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override results directory. Defaults to testbed/results/.",
)
def benchmark_cmd(config_path: Path, runs_root: Path | None) -> None:
    """Run a set of configured (challenge, interface, skills) combinations once."""
    from banter import benchmark as bm_mod

    config = bm_mod.load_config(config_path.resolve())
    rr = (runs_root or TESTBED_ROOT / "results").resolve()
    bm_mod.run_benchmark(config, rr)


# ---------------------------------------------------------------------------
# banter interfaces / skills
# ---------------------------------------------------------------------------


@main.command("interfaces")
def list_interfaces() -> None:
    """List configured interfaces and their types."""
    for name, types in interfaces.available().items():
        click.echo(f"{name}: {', '.join(types) if types else '(no types)'}")


@main.command("skills")
def list_skills() -> None:
    """List available skill bundles under configs/skills/."""
    for bundle in skills.available():
        click.echo(bundle)


# ---------------------------------------------------------------------------
# banter setup
# ---------------------------------------------------------------------------


@main.command("setup")
def setup() -> None:
    """Interactive credential + auth setup. Idempotent — re-run to update."""
    click.echo("== banter setup ==")
    click.echo(f"Claude Code default model: {runner.DEFAULT_MODEL}")
    _setup_kaggle()
    _setup_claude_auth()
    click.echo("\nDone. Try: banter run --challenge titanic --interface none")


def _setup_kaggle() -> None:
    click.secho("\n[1/2] Kaggle credentials", bold=True)
    click.echo("  Required by mle-bench to download competition data.")
    click.echo("  IMPORTANT: use the LEGACY API key (32 hex chars), not the new")
    click.echo("  access token (starts with 'KGAT...'). Get one at:")
    click.echo("    https://www.kaggle.com/settings  →  API  →  Create New API Token")
    click.echo("  which downloads a kaggle.json containing {username, key}.")
    kaggle_json = KAGGLE_DIR / "kaggle.json"
    if kaggle_json.exists():
        click.echo(f"  Existing credentials at {kaggle_json}.")
        if not click.confirm("  Overwrite?", default=False):
            return
    username = click.prompt("  Kaggle username")
    key = click.prompt("  Kaggle API key (legacy, 32 hex chars)", hide_input=True)
    if key.startswith("KGAT") or len(key) != 32:
        click.secho(
            f"  Warning: key looks unusual (length={len(key)}, prefix={key[:4]!r}). "
            "The legacy key is 32 hex chars. Proceeding anyway.",
            fg="yellow",
        )
    kaggle_json.parent.mkdir(parents=True, exist_ok=True)
    kaggle_json.write_text(json.dumps({"username": username, "key": key}))
    kaggle_json.chmod(0o600)
    click.echo(f"  Wrote {kaggle_json}")


def _setup_claude_auth() -> None:
    click.secho("\n[2/2] Claude Code auth", bold=True)
    click.echo("  api-key  — claude-code calls Anthropic directly with ANTHROPIC_API_KEY.")
    click.echo("  login    — claude-code uses your Claude subscription via `claude /login`.")
    click.echo("  Tokens + cost are read from claude-code's own transcript in either mode.")
    choice = click.prompt(
        "  Auth mode",
        type=click.Choice(list(runner.AUTH_MODES)),
        default="api-key",
    )
    if choice == "api-key":
        api_key = click.prompt("  Anthropic API key", hide_input=True)
        _write_dotenv({"BANTER_AUTH": "api-key", "ANTHROPIC_API_KEY": api_key})
        click.echo(f"  Wrote ANTHROPIC_API_KEY to {DOTENV_PATH}")
    else:
        _write_dotenv({"BANTER_AUTH": "login"})
        if shutil.which("claude") is None:
            click.secho("  `claude` CLI not on PATH — install Claude Code first.", fg="yellow")
            return
        if click.confirm("  Launch `claude /login` now?", default=True):
            subprocess.run(["claude", "/login"], check=False)


if __name__ == "__main__":
    main()
