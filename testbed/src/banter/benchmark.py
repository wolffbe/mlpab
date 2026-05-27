"""One-shot benchmark runner.

Executes a list of explicitly specified runs and prints a summary table.
Each run can pin exact interface and skills versions so you can reproduce or
compare any configuration produced during autoresearch.

Results are isolated per session:
    runs/benchmark/<session_id>/results.csv
    runs/benchmark/<session_id>/<run_dir>/

Config format (YAML):

    engineer_model: claude-sonnet-4-6   # model for the engineer (legacy: model)
    engineer_auth: api-key              # engineer auth (legacy: auth)

    # Option A — explicit run list (supports session/version pinning)
    runs:
      - challenge: aerial-cactus-identification
        interface: none
        mode: none
        skills: none
      - challenge: aerial-cactus-identification
        interface: hopsworks
        mode: cli
        # Interface versions live inside an autoresearch session — pin both:
        session: a1b2c3d4         # the autoresearch session that produced it
        interface_version: 2      # the version number within that session
        skills: none

    # Option B — Cartesian matrix
    # challenges: [aerial-cactus-identification]
    # interfaces: [{name: hopsworks, mode: cli, session: a1b2c3d4, version: 2}]
    # skills: [none]

    timeout_s: 3600    # per-run wall-clock cap (optional)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from banter import interfaces, preflight as preflight_mod, results, runner


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
    # Autoresearch session that produced the pinned interface version (needed
    # to resolve interface_version > 0, which lives inside that session).
    session: str | None = None


@dataclass
class BenchmarkConfig:
    runs: list[RunEntry]
    engineer_model: str = runner.DEFAULT_MODEL
    engineer_auth: str = "api-key"
    timeout_s: int = 3600


def _parse_runs(data: dict[str, Any]) -> list[RunEntry]:
    """Return the run list from the config dict.

    Supports both an explicit `runs:` list and a Cartesian matrix via
    `challenges`, `interfaces`, and `skills` keys.
    """
    if "runs" in data:
        entries = []
        for r in data["runs"]:
            if r.get("config"):
                name, mode = interfaces.name_type_from_config(r["config"])
            else:
                name, mode = r["interface"], r.get("mode", "none")
            entries.append(
                RunEntry(
                    challenge=r["challenge"],
                    interface=name,
                    mode=mode,
                    skills=r.get("skills", "none"),
                    interface_version=r.get("interface_version", r.get("version")),
                    skills_version=r.get("skills_version"),
                    session=r.get("session"),
                )
            )
        return entries

    # Cartesian matrix fallback. Each interface entry references its config
    # (`config:`) or gives name+mode, and may pin a `version:` (+ `session:`).
    challenges = data.get("challenges") or []
    iface_entries = data.get("interfaces") or []
    skills_list = data.get("skills") or ["none"]
    entries = []
    for ch in challenges:
        for iface in iface_entries:
            if not isinstance(iface, dict):
                raise ValueError(
                    f"benchmark `interfaces` entries must be mappings, got {iface!r}"
                )
            if iface.get("config"):
                name, mode = interfaces.name_type_from_config(iface["config"])
            else:
                name, mode = iface["name"], iface.get("mode", "none")
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
                        interface=name,
                        mode=mode,
                        skills=sk_name,
                        interface_version=iface.get("version"),
                        skills_version=sk_version,
                        session=iface.get("session"),
                    )
                )
    return entries


def load_config(path: Path) -> BenchmarkConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return BenchmarkConfig(
        runs=_parse_runs(data),
        # `engineer_model`/`engineer_auth` are the canonical keys; `model`/`auth`
        # are accepted as legacy fallbacks so older configs keep working.
        engineer_model=data.get("engineer_model", data.get("model", runner.DEFAULT_MODEL)),
        engineer_auth=data.get("engineer_auth", data.get("auth", "api-key")),
        timeout_s=int(data.get("timeout_s", 3600)),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _version_root_for(session: str | None) -> Path | None:
    """Session dir holding pinned interface versions, or None for the base."""
    if not session:
        return None
    return runner.TESTBED_ROOT / "results" / "autoresearch" / session


def run_benchmark(config: BenchmarkConfig, runs_root: Path) -> None:
    total = len(config.runs)
    if total == 0:
        print("[benchmark] No runs configured.", file=sys.stderr)
        return

    # Fail fast, once, over the union of requirements before doing any work.
    reqs = [
        preflight_mod.Requirement(
            interface=e.interface,
            mode=e.mode,
            interface_version=e.interface_version,
            version_root=_version_root_for(e.session),
            skills=e.skills,
            skills_version=e.skills_version,
        )
        for e in config.runs
    ]
    try:
        preflight_mod.preflight(reqs, auth=config.engineer_auth, model=config.engineer_model)
    except preflight_mod.PreflightError as e:
        print(f"\n[benchmark] preflight failed:\n{e}", file=sys.stderr)
        raise

    parent = runs_root / "benchmark"
    session_id = results.next_session_id(parent)
    session_dir = parent / session_id
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
                model=config.engineer_model,
                auth=config.engineer_auth,
                timeout_s=config.timeout_s,
                runs_root=session_dir,
                interface_version=entry.interface_version,
                skills_version=entry.skills_version,
                version_root=_version_root_for(entry.session),
                preflight=False,  # union already verified upfront
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

    # High-level rollup: one aggregated row per session at results/benchmark/results.csv.
    if completed:
        ifaces = sorted({f"{r.interface}/{r.mode}" for r in completed})
        skills = sorted({r.skills for r in completed})
        results.append_summary(
            parent / "results.csv",
            results.BENCHMARK_SUMMARY_FIELDS,
            {
                "session": session_id,
                "started_at": completed[0].started_at,
                "interfaces": "|".join(ifaces),
                "skills": "|".join(skills),
                "n_runs": len(completed),
                "avg_score": results._avg([r.score for r in completed]),
                "avg_total_tokens": results._avg([r.total_tokens for r in completed]),
                "avg_wall_time_s": results._avg([r.wall_time_s for r in completed]),
                "avg_cost_usd": results._avg([r.cost_usd for r in completed]),
            },
        )

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
