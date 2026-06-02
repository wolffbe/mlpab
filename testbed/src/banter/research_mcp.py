"""Researcher MCP server — a single `normalized_composite` tool.

Launched alongside the autoresearch researcher (registered via the run dir's
`.mcp.json`). Gives the researcher a deterministic way to score the versions it
has created: it reads the global experiments table, filters to THIS treatment
(by config path), mean-aggregates each metric across the treatment's challenges
per version, and returns the normalized composite J per version plus the best
version — using the goal metrics/directions set in the treatment's config.

Deliberately dependency-free: MCP's stdio transport is newline-delimited
JSON-RPC 2.0, which we speak directly so this needs no `mcp`/`fastmcp` package
(those aren't in banter's environment, and adding a dependency needs sign-off).

Configuration via env (set by `autoresearch.run_autoresearch`):
  BANTER_EXPERIMENTS_CSV  path to results/experiments.csv
  BANTER_CONFIG           treatment config path (the row's identity / filter)
  BANTER_GOALS            goals string, e.g. "python_calls:min|score:max"
"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any

from banter import experiments

PROTOCOL_VERSION = "2024-11-05"

_TOOL = {
    "name": "normalized_composite",
    "description": (
        "Score the versions created so far for THIS treatment. Reads the global "
        "experiments table, mean-aggregates each metric across the treatment's "
        "challenges per version, and returns — for ALL versions so far (v0 "
        "baseline through the latest, the full history) — the composite J, PLUS "
        "each optimization goal's value, direction, and normalized contribution "
        "to J, PLUS every observed (non-goal) metric. Use it to compare against "
        "previous versions, see which is best, and see what is driving the score "
        "so you know what to optimize next: a low normalized contribution on a "
        "goal is where to push."
    ),
    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
}

_SHORT = {"minimize": "min", "maximize": "max"}


def _compute() -> dict[str, Any]:
    csv_path = os.environ.get("BANTER_EXPERIMENTS_CSV", "")
    config_rel = os.environ.get("BANTER_CONFIG", "")
    platform = os.environ.get("BANTER_PLATFORM", "")
    interface = os.environ.get("BANTER_INTERFACE", "")
    goals = experiments.parse_goals_str(os.environ.get("BANTER_GOALS", ""))
    if not csv_path or not os.path.exists(csv_path):
        return {"error": "experiments table not found", "versions": {}, "best_version": None}

    def _match(r: dict) -> bool:
        # Scope to THIS leaf: a config may span several interfaces.
        if r.get("config") != config_rel:
            return False
        if platform and r.get("platform") != platform:
            return False
        if interface and r.get("interface") != interface:
            return False
        return True

    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if _match(r)]
    goal_str = [f"{m}:{_SHORT.get(d, d)}" for m, d in goals]
    if not rows:
        return {"goals": goal_str, "versions": {}, "best_version": None,
                "note": "no runs recorded for this treatment yet"}

    per = experiments.aggregate_by_version(rows, experiments.METRICS)
    breakdown = experiments.composite_breakdown(per, goals)
    best = experiments.best_version(per, goals)
    goal_metrics = [m for m, _ in goals]
    dir_of = {m: _SHORT.get(d, d) for m, d in goals}

    versions: dict[str, Any] = {}
    for v in sorted(per):
        contrib = breakdown[v]["contributions"]
        versions[f"v{v}"] = {
            "composite": breakdown[v]["composite"],
            # The optimization targets: value, direction, and how much this
            # version's value contributes to J (0 = worst on this goal, 1 = best).
            "goals": {
                m: {
                    "value": per[v].get(m),
                    "direction": dir_of[m],
                    "normalized": contrib.get(m),
                }
                for m in goal_metrics
            },
            # Everything else, observed (not optimized) — context for tradeoffs.
            "observed": {
                m: per[v].get(m) for m in experiments.METRICS if m not in goal_metrics
            },
        }
    return {"goals": goal_str, "best_version": f"v{best}" if best is not None else None,
            "versions": versions}


def _result(rid: Any, payload: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": payload}


def _handle(msg: dict) -> dict | None:
    method = msg.get("method")
    rid = msg.get("id")
    if method == "initialize":
        return _result(rid, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "banter-research", "version": "0.1.0"},
        })
    if method in ("notifications/initialized", "initialized"):
        return None  # notification — no response
    if method == "tools/list":
        return _result(rid, {"tools": [_TOOL]})
    if method == "tools/call":
        name = (msg.get("params") or {}).get("name")
        if name != _TOOL["name"]:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32601, "message": f"unknown tool: {name}"}}
        try:
            text = json.dumps(_compute(), indent=2)
            return _result(rid, {"content": [{"type": "text", "text": text}]})
        except Exception as e:  # surface as a tool error, not a crash
            return _result(rid, {
                "content": [{"type": "text", "text": f"error: {e}"}],
                "isError": True,
            })
    if rid is not None:
        return {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601, "message": f"method not found: {method}"}}
    return None


def main() -> None:
    """Read newline-delimited JSON-RPC from stdin, write responses to stdout."""
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = _handle(msg)
        if resp is not None:
            out.write(json.dumps(resp) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
