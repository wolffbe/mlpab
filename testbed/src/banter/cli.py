"""banter CLI.

Examples:
    banter setup                                                      # one-time creds setup
    banter run --challenge titanic --interface none
    banter run --challenge titanic --interface hopsworks --mode sdk
    banter run --challenge titanic --interface hopsworks --mode cli --skills hopsworks-essentials
    banter interfaces                                                 # list configured interfaces / types
    banter skills                                                     # list available skill bundles
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import click

from banter import interfaces, runner, skills


# Testbed repo root. We anchor all per-project state (.env, Kaggle creds,
# mle-bench cache) here so the repo is self-contained regardless of where
# `banter` is invoked from.
TESTBED_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = TESTBED_ROOT / ".env"
KAGGLE_DIR = TESTBED_ROOT / ".kaggle"


def _load_dotenv() -> None:
    # Point Kaggle CLI / SDK at our in-repo creds dir before anything else
    # imports kaggle (which captures KAGGLE_CONFIG_DIR at import time).
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


@main.command()
@click.option("--challenge", required=True, help="MLE-bench competition id, e.g. `titanic`.")
@click.option(
    "--interface",
    "interface_name",
    required=True,
    help="Interface NAME from config/interfaces.yaml (e.g. `hopsworks`, `none`).",
)
@click.option(
    "--mode",
    type=click.Choice(interfaces.TYPES),
    default=None,
    help="Interface TYPE: cli, mcp, sdk, or none. Defaults to `none` when --interface=none.",
)
@click.option(
    "--skills",
    default="none",
    show_default=True,
    help="Skill bundle name under testbed/skills/, or `none`.",
)
@click.option("--model", default=runner.DEFAULT_MODEL, show_default=True)
@click.option(
    "--auth",
    type=click.Choice(runner.AUTH_MODES),
    # Default to whatever `banter setup` saved; fall back to api-key.
    default=lambda: os.environ.get("BANTER_AUTH", "api-key"),
    show_default="from .env or api-key",
    help="api-key: claude-code uses ANTHROPIC_API_KEY from env / .env. "
    "login: claude-code uses stored OAuth credentials (run `claude /login` first).",
)
@click.option("--timeout", "timeout_s", type=int, default=60 * 60, show_default=True, help="Wall-clock cap for claude -p.")
@click.option("--runs-root", type=click.Path(path_type=Path), default=Path("runs"), show_default=True)
def run(
    challenge: str,
    interface_name: str,
    mode: str | None,
    skills: str,
    model: str,
    auth: str,
    timeout_s: int,
    runs_root: Path,
) -> None:
    """Run one challenge and append a row to runs/results.csv."""
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
    )
    row = runner.run(spec)
    click.echo(
        f"\n=== {challenge} [{row.interface}/{row.mode}, skills={row.skills}] "
        f"done in {row.wall_time_s:.1f}s "
        f"({row.total_tokens} tokens, ${row.cost_usd:.4f}); medal={row.medal} ===\n"
        f"run dir: {row.run_dir}"
    )


@main.command("interfaces")
def list_interfaces() -> None:
    """List configured interfaces and their types."""
    for name, types in interfaces.available().items():
        click.echo(f"{name}: {', '.join(types) if types else '(no types)'}")


@main.command("skills")
def list_skills() -> None:
    """List available skill bundles under testbed/skills/."""
    for bundle in skills.available():
        click.echo(bundle)


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
