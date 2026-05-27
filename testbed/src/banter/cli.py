"""banter CLI — deliberately small.

Two things only:

    banter <config.yaml>     run an autoresearch or benchmark config
    banter setup [MANIFEST]   one-time auth + credential keys (used by `make setup`)

Everything else — building interface binaries, logging in, testing they run —
happens automatically at preflight when you run a config, with no AI involved.
The researcher drives individual challenge evaluations internally via the
`run --challenge ...` form (not part of the everyday surface).

Examples:
    banter configs/autoresearch_rq1_hopsworks.yaml
    banter configs/benchmark_smoke_test.yaml
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
@click.option("--interface", "interface_name", default=None, help="Interface name, e.g. `hopsworks`, `none`.")
@click.option("--mode", type=click.Choice(interfaces.TYPES), default=None, help="Interface type.")
@click.option("--interface-version", "interface_version", type=int, default=None,
              help="Interface version. Omit/0 → base config; >0 → a session version.")
@click.option("--version-root", "version_root", type=click.Path(path_type=Path), default=None,
              help="Session dir holding versions > 0 (required when --interface-version > 0).")
@click.option("--skills", default="none", show_default=True, help="Skill bundle name under skills/, or `none`.")
@click.option("--skills-version", "skills_version", type=int, default=None, help="Pin a skill bundle version.")
@click.option("--model", default=runner.DEFAULT_MODEL, show_default=True)
@click.option("--auth", type=click.Choice(runner.AUTH_MODES),
              default=lambda: os.environ.get("BANTER_AUTH", "api-key"), show_default="from .env or api-key")
@click.option("--timeout", "timeout_s", type=int, default=60 * 60, show_default=True)
@click.option("--runs-root", type=click.Path(path_type=Path), default=Path("results"), show_default=True)
def run(
    config: str | None,
    challenge: str | None,
    interface_name: str | None,
    mode: str | None,
    interface_version: int | None,
    version_root: Path | None,
    skills: str,
    skills_version: int | None,
    model: str,
    auth: str,
    timeout_s: int,
    runs_root: Path,
) -> None:
    """Run an autoresearch/benchmark CONFIG, or a single challenge."""
    if config is not None:
        _dispatch_config(Path(config), runs_root)
        return

    if challenge is None:
        raise click.UsageError("Pass a CONFIG file (e.g. `banter configs/...yaml`) or use --challenge.")
    if interface_name is None:
        raise click.UsageError("--interface is required for the single-challenge form.")
    if mode is None:
        mode = "none" if interface_name == "none" else None
        if mode is None:
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
        version_root=version_root,
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
# banter setup  (auth for engineer + researcher; credential keys per interface)
# ---------------------------------------------------------------------------


@main.command("setup")
@click.argument("manifest", required=False, type=click.Path(path_type=Path))
def setup(manifest: Path | None) -> None:
    """One-time setup. With no argument: Kaggle + Claude auth + all interface keys.
    With a MANIFEST: set just that interface's credential keys."""
    if manifest is not None:
        _setup_interface_keys(manifest.resolve())
        return
    click.echo("== banter setup ==")
    click.echo(f"Claude Code default model: {runner.DEFAULT_MODEL}")
    _setup_kaggle()
    _setup_claude_auth()
    _setup_all_interface_keys()
    click.echo("\nDone. Try: banter configs/benchmark_smoke_test.yaml")


def _detect_name_type(config_path: Path) -> tuple[str, str]:
    """Infer (name, type) from configs/interfaces/<name>/<type>.yaml."""
    parts = config_path.parts
    try:
        idx = list(parts).index("interfaces")
        name = parts[idx + 1]
        type_ = Path(parts[idx + 2]).stem
    except (ValueError, IndexError):
        raise click.ClickException(
            f"Cannot infer interface name/type from {config_path}. "
            "Expected: configs/interfaces/<name>/<type>.yaml"
        )
    if type_ not in interfaces.TYPES:
        raise click.ClickException(f"Unknown interface type {type_!r}; expected one of {interfaces.TYPES}")
    return name, type_


def _setup_kaggle() -> None:
    click.secho("\n[1/3] Kaggle credentials", bold=True)
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
    click.secho("\n[2/3] Claude Code auth (engineer + researcher)", bold=True)
    click.echo("  Both the engineer (controlled) and researcher (controller) instances")
    click.echo("  authenticate the same way:")
    click.echo("    api-key  — claude-code calls Anthropic with ANTHROPIC_API_KEY.")
    click.echo("    login    — claude-code uses your Claude subscription via `claude /login`.")
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
        if click.confirm("  Launch `claude /login` now? (logs in engineer + researcher)", default=True):
            subprocess.run(["claude", "/login"], check=False)


def _setup_all_interface_keys() -> None:
    click.secho("\n[3/3] Interface credential keys", bold=True)
    avail = interfaces.available()
    any_keys = False
    for name, types in avail.items():
        for type_ in types:
            if interfaces.keys_for(name, type_):
                any_keys = True
                mpath = interfaces.manifest_path(name, type_)
                if click.confirm(f"  Set keys for {name}/{type_}?", default=True):
                    _setup_interface_keys(mpath)
    if not any_keys:
        click.echo("  No interfaces declare `keys:` — nothing to set.")


def _setup_interface_keys(manifest_path: Path) -> None:
    if not manifest_path.exists():
        raise click.ClickException(f"Manifest not found: {manifest_path}")
    name, type_ = _detect_name_type(manifest_path)
    keys = interfaces.keys_for(name, type_)
    if not keys:
        click.echo(f"  {name}/{type_}: no `keys:` declared in the config.")
        return
    click.secho(f"  Credentials for {name}/{type_}:", bold=True)
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
