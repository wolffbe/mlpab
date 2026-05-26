"""One-shot benchmark runner.

Executes a list of explicitly specified runs and prints a summary table.
Each run can pin exact interface and skills versions so you can reproduce or
compare any configuration produced during autoresearch.

Results are isolated per session:
    runs/benchmark/<session_id>/results.csv
    runs/benchmark/<session_id>/<run_dir>/

Config format (YAML):

    model: claude-sonnet-4-6
    auth: api-key

    # Option A — explicit run list (supports version pinning)
    runs:
      - challenge: aerial-cactus-identification
        interface: none
        mode: none
        # interface_version: 0   # omit → latest
        skills: none
        # skills_version: null   # omit → latest
      - challenge: aerial-cactus-identification
        interface: none
        mode: none
        interface_version: 0     # run an old autoresearch-generated version
        skills: none

    # Option B — Cartesian matrix (all at latest versions)
    # challenges: [aerial-cactus-identification]
    # interfaces: [{name: none, mode: none}]
    # skills: [none]

    timeout_s: 3600    # per-run wall-clock cap (optional)
"""
from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from banter import results, runner


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RunEntry:
    challenge: str
    interface: str
    mode: str
    skills: str = "none"
    interface_version: int | None = None
    skills_version: int | None = None


@dataclass
class BenchmarkConfig:
    runs: list[RunEntry]
    model: str = runner.DEFAULT_MODEL
    auth: str = "api-key"
    timeout_s: int = 3600


def _parse_runs(data: dict[str, Any]) -> list[RunEntry]:
    """Return the run list from the config dict.

    Supports both an explicit `runs:` list and a Cartesian matrix via
    `challenges`, `interfaces`, and `skills` keys.
    """
    if "runs" in data:
        entries = []
        for r in data["runs"]:
            entries.append(
                RunEntry(
                    challenge=r["challenge"],
                    interface=r["interface"],
                    mode=r.get("mode", "none"),
                    skills=r.get("skills", "none"),
                    interface_version=r.get("interface_version"),
                    skills_version=r.get("skills_version"),
                )
            )
        return entries

    # Cartesian matrix fallback. Each interface and skill entry may pin a
    # specific `version:` integer; omit it to use the latest.
    challenges = data.get("challenges") or []
    interfaces = data.get("interfaces") or []
    skills_list = data.get("skills") or ["none"]
    entries = []
    for ch in challenges:
        for iface in interfaces:
            if not isinstance(iface, dict):
                raise ValueError(
                    f"benchmark `interfaces` entries must be mappings, got {iface!r}"
                )
            for sk in skills_list:
                if isinstance(sk, str):
                    sk_name, sk_version = sk, None
                elif isinstance(sk, dict):
                    sk_name = sk.get("name", "none")
                    sk_version = sk.get("version")
                else:
                    raise ValueError(f"benchmark `skills` entry must be str or mapping, got {sk!r}")
                entries.append(
                    RunEntry(
                        challenge=ch,
                        interface=iface["name"],
                        mode=iface.get("mode", "none"),
                        skills=sk_name,
                        interface_version=iface.get("version"),
                        skills_version=sk_version,
                    )
                )
    return entries


def load_config(path: Path) -> BenchmarkConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return BenchmarkConfig(
        runs=_parse_runs(data),
        model=data.get("model", runner.DEFAULT_MODEL),
        auth=data.get("auth", "api-key"),
        timeout_s=int(data.get("timeout_s", 3600)),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_benchmark(config: BenchmarkConfig, runs_root: Path) -> None:
    total = len(config.runs)
    if total == 0:
        print("[benchmark] No runs configured.", file=sys.stderr)
        return

    session_id = uuid.uuid4().hex[:8]
    session_dir = runs_root / "benchmark" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"[benchmark] session={session_id}  runs={total}  dir={session_dir}")

    completed: list[results.Row] = []
    failed: list[str] = []

    for n, entry in enumerate(config.runs, 1):
        version_tag = f" v{entry.interface_version}" if entry.interface_version is not None else ""
        skills_tag = entry.skills
        if entry.skills_version is not None:
            skills_tag += f" v{entry.skills_version}"
        print(
            f"\n[benchmark {n}/{total}] {entry.challenge} | "
            f"{entry.interface}/{entry.mode}{version_tag} | skills={skills_tag}"
        )
        try:
            spec = runner.RunSpec(
                challenge_id=entry.challenge,
                interface=entry.interface,
                mode=entry.mode,
                skills=entry.skills,
                model=config.model,
                auth=config.auth,
                timeout_s=config.timeout_s,
                runs_root=session_dir,
                interface_version=entry.interface_version,
                skills_version=entry.skills_version,
            )
            row = runner.run(spec)
            completed.append(row)
            print(
                f"[benchmark] score={row.score}  tokens={row.total_tokens}  "
                f"wall={row.wall_time_s:.1f}s  cost=${row.cost_usd:.4f}"
            )
        except Exception as exc:
            label = f"{entry.challenge}/{entry.interface}/{entry.mode}"
            print(f"[benchmark] FAILED {label}: {exc}", file=sys.stderr)
            failed.append(f"{label}: {exc}")

    _print_summary(completed, failed, session_dir)


def _print_summary(rows: list[results.Row], failed: list[str], session_dir: Path) -> None:
    w = 110
    print("\n" + "=" * w)
    print("BENCHMARK SUMMARY")
    print("=" * w)
    if rows:
        print(
            f"{'challenge':<35} {'iface/mode':<20} {'ver':<4} {'skills':<18} "
            f"{'score':<8} {'tokens':<8} {'wall_s':<7} cost"
        )
        print("-" * w)
        for r in rows:
            print(
                f"{r.challenge_id:<35} {r.interface}/{r.mode:<18} {r.version:<4} {r.skills:<18} "
                f"{str(r.score or ''):<8} {r.total_tokens:<8} {r.wall_time_s:<7.1f} ${r.cost_usd:.4f}"
            )
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")
    print(f"\nResults CSV: {session_dir}/results.csv")
    print("=" * w)
