"""Per-treatment autoresearch configs → a global results table + analysis notebook.

The autoresearch **config files are the source of truth** — one hand-edited
config per treatment under `platforms/<platform>/autoresearch/…`.

`results/autoresearch/experiments.csv` is the **global, factual overview** — NOT
computed. The runner appends one EXPLODED row per `(treatment, version, task,
challenge)` LIVE via `append_run()` (design metadata + the raw engineer metrics);
`attribute_researcher()` fills the researcher's share at session end, and
`clear_treatment()` drops a treatment's old rows when it is re-run. Only
treatments that have actually run appear.

All derived math — the normalized composite, the best version per treatment,
baseline-vs-best deltas, cross-treatment aggregation and charts — lives in the
single global `results/autoresearch/analysis.ipynb` (`build_global_notebook()`),
which reuses `aggregate_by_version` / `normalized_composite` / `best_version`.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Numeric per-run metric columns. `score`/`python_calls`/`total_tokens` are the
# engineer-side optimization metrics (what the goals reference). Wall time and
# cost are split engineer / researcher / total: the runner writes the engineer
# side live; the researcher's shared overhead is distributed across the leaf's
# rows at session end (`attribute_researcher`), and `total_* = eng_* + res_*`.
METRICS = [
    "score",
    # Tokens — engineer / researcher / total, each with input·output·total.
    "eng_input_tokens", "eng_output_tokens", "eng_total_tokens",
    "res_input_tokens", "res_output_tokens", "res_total_tokens",
    "total_input_tokens", "total_output_tokens", "total_tokens",
    # Wall time — engineer / researcher / total.
    "eng_wall_time_s", "res_wall_time_s", "total_wall_time_s",
    # Cost (USD) — engineer / researcher / total.
    "eng_cost_usd", "res_cost_usd", "total_cost",
    # Tool-call counts (python_calls follows llm_calls).
    "llm_calls", "python_calls", "cli_calls", "mcp_calls", "sdk_calls",
    "bash_calls", "skill_calls", "other_tool_calls",
]

# Engineer columns (written live by the runner) and their researcher / total
# counterparts, paired for the session-end attribution (total = eng + res).
_ATTR_PAIRS = [
    ("eng_input_tokens", "res_input_tokens", "total_input_tokens"),
    ("eng_output_tokens", "res_output_tokens", "total_output_tokens"),
    ("eng_total_tokens", "res_total_tokens", "total_tokens"),
    ("eng_wall_time_s", "res_wall_time_s", "total_wall_time_s"),
    ("eng_cost_usd", "res_cost_usd", "total_cost"),
]
# Researcher per-row share key → the usage dict key it's distributed from.
_RES_FROM = {
    "res_input_tokens": "input_tokens",
    "res_output_tokens": "output_tokens",
    "res_total_tokens": "total_tokens",
    "res_wall_time_s": "wall_s",
    "res_cost_usd": "cost_usd",
}

# Treatment-level design columns (repeated on every exploded row). `goals` and
# `budget` are JSON arrays (e.g. ["python_calls:min","score:max"] and
# ["iterations:4","max_seconds:28800"]). `config` is the treatment's identity.
DESIGN_FIELDS = [
    "experiment",
    "research_question",
    "treatment",
    "config",
    "researcher_model",
    "engineer_model",
    "platform",
    "interface",
    "language",
    "skills",
    "optimization_variable",
    "goals",
    "time",
    "budget",
]

# The exploded grain + per-row run values. `started_at` = when the engineer's
# `claude -p` for this (version, challenge) began.
GRAIN_FIELDS = ["version", "task", "challenge", "started_at"]

# Per-version researcher annotations (filled by `banter annotate-version` after a
# version's runs finish). They are denormalized — the same values land on every
# challenge row of that (treatment, version) — so a single CSV is the full
# record. `verdict` ∈ positive|negative|neutral; `keep` ∈ 0|1.
ANNOTATION_FIELDS = [
    "hypothesis", "change", "verdict", "verdict_reason",
    "keep", "observations", "proposed_changes",
]
RESULT_FIELDS = METRICS + ["valid_submission", "medal"] + ANNOTATION_FIELDS + ["error", "run_dir"]

EXPERIMENT_FIELDS = DESIGN_FIELDS + GRAIN_FIELDS + RESULT_FIELDS


# ---------------------------------------------------------------------------
# Shared composite math (used by the global notebook)
# ---------------------------------------------------------------------------


def _fnum(x: Any) -> float | None:
    try:
        s = str(x).strip()
        return float(s) if s != "" else None
    except (TypeError, ValueError):
        return None


def _to_int_version(v: Any) -> int:
    s = str(v).strip().lstrip("v")
    return int(s) if s.isdigit() else -1


def aggregate_by_version(rows: list[dict], metrics: list[str]) -> dict[int, dict[str, float | None]]:
    """Mean each metric across a version's challenge rows → {version: {metric: mean}}."""
    buckets: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        v = _to_int_version(r.get("version", ""))
        if v < 0:
            continue
        b = buckets.setdefault(v, {m: [] for m in metrics})
        for m in metrics:
            val = _fnum(r.get(m))
            if val is not None:
                b[m].append(val)
    return {
        v: {m: (sum(vals) / len(vals) if vals else None) for m, vals in b.items()}
        for v, b in buckets.items()
    }


def composite_breakdown(
    per_version: dict[int, dict[str, float | None]],
    goals: list[tuple[str, str]],
) -> dict[int, dict[str, Any]]:
    """Per version: the composite J plus each goal's normalized contribution.

    Each goal metric is min-max normalized across versions to [0, 1] (1 = best
    in its direction). The composite is the equal-weight sum of contributions.
    A goal with no spread contributes 0 to every version. Exposing the per-goal
    contributions lets the researcher see WHAT is driving the score.
    """
    versions = sorted(per_version)
    contrib: dict[int, dict[str, float]] = {v: {} for v in versions}
    for metric, direction in goals:
        vals = {v: per_version[v].get(metric) for v in versions}
        nums = [x for x in vals.values() if x is not None]
        if not nums:
            continue
        lo, hi = min(nums), max(nums)
        span = hi - lo
        for v in versions:
            x = vals[v]
            if x is None or span == 0:
                # Missing value → worst (0) so every version sums the SAME goals
                # (composites stay comparable). No spread → 0 for all (no signal).
                contrib[v][metric] = 0.0
                continue
            norm = (x - lo) / span                # 0..1, higher = larger
            if direction == "minimize":
                norm = 1.0 - norm                 # higher = better
            contrib[v][metric] = round(norm, 6)
    return {
        v: {"composite": round(sum(contrib[v].values()), 6), "contributions": contrib[v]}
        for v in versions
    }


def normalized_composite(
    per_version: dict[int, dict[str, float | None]],
    goals: list[tuple[str, str]],
) -> dict[int, float]:
    """Composite objective J(version), higher = better — the equal-weight sum of
    each goal's normalized contribution (see `composite_breakdown`)."""
    return {v: b["composite"] for v, b in composite_breakdown(per_version, goals).items()}


def best_version(
    per_version: dict[int, dict[str, float | None]],
    goals: list[tuple[str, str]],
) -> int | None:
    """The version maximizing the normalized composite. Ties break to the LOWER
    version: when no goal separates the versions (e.g. all metrics flat, or only
    a v0 baseline), the honest "best" is the baseline, not a later no-op version."""
    if not per_version:
        return None
    J = normalized_composite(per_version, goals)
    return max(per_version, key=lambda v: (J[v], -v))


# ---------------------------------------------------------------------------
# Spec → treatments → configs
# ---------------------------------------------------------------------------


def _skills_label(improve: list[str], skills: str | None) -> str:
    if "skills" in (improve or []):
        return "yes"
    if skills and skills != "none":
        return "yes"
    return "no"


def _dir_label(direction: str) -> str:
    return {"minimize": "min", "maximize": "max"}.get(direction, "obs")


def _goals_str(goals: list[tuple[str, str]]) -> str:
    """JSON array, e.g. [(python_calls, minimize), (score, maximize)] →
    '["python_calls:min", "score:max"]'."""
    return json.dumps([f"{m}:{_dir_label(d)}" for m, d in goals])


def parse_goals_str(s: str) -> list[tuple[str, str]]:
    """Inverse of `_goals_str`. Accepts the JSON-array form and the legacy
    pipe-joined form ('python_calls:min|score:max')."""
    s = (s or "").strip()
    if not s:
        return []
    items: list[str]
    if s.startswith("["):
        try:
            items = [str(x) for x in json.loads(s)]
        except (ValueError, TypeError):
            items = []
    else:
        items = s.split("|")
    out: list[tuple[str, str]] = []
    for tok in items:
        tok = tok.strip()
        if ":" not in tok:
            continue
        m, d = tok.split(":", 1)
        out.append((m, "minimize" if d == "min" else "maximize"))
    return out


def _budget_arr(budget: "Any") -> str:
    """JSON array of the budgets in force, e.g. ["iterations:4","max_seconds:28800"].
    Infinite caps are omitted."""
    parts = [f"iterations:{budget.iterations}"]
    if budget.max_seconds != float("inf"):
        parts.append(f"max_seconds:{int(budget.max_seconds)}")
    if budget.max_cost_usd != float("inf"):
        parts.append(f"max_cost_usd:{budget.max_cost_usd}")
    return json.dumps(parts)


_DELEGATION_METRICS = {"cli_calls", "mcp_calls", "sdk_calls"}


_OPTVAR_BY_COUNT = {1: "univariate", 2: "bivariate", 3: "multivariate"}


def _optvar_for_count(n: int) -> str:
    """The optimization-variable label for a goal count: 1→univariate,
    2→bivariate, 3→multivariate; "" for 0 or >3."""
    return _OPTVAR_BY_COUNT.get(n, "")


def goals_for_interface(goals: list[tuple[str, str]], interface: str) -> list[tuple[str, str]]:
    """Drop delegation goals (`cli_calls`/`mcp_calls`/`sdk_calls`) that don't
    match `interface`. An engineer uses ONE interface, so only that interface's
    call count is achievable — maximizing all three at once is impossible. The
    interface's own `*_calls` goal is kept; the others are removed. Non-delegation
    goals (score, tokens, python_calls, …) pass through unchanged."""
    keep = f"{interface}_calls"
    return [(m, d) for (m, d) in goals if m not in _DELEGATION_METRICS or m == keep]


def _config_rel_path(platform: str, rq: int, treatment: int, interface: str, opt_var: str) -> str:
    return (
        f"platforms/{platform}/autoresearch/rq{rq}"
        f"/t{treatment:02d}-{interface}-{opt_var}.yaml"
    )


# ---------------------------------------------------------------------------
# Global registry (scan configs = truth + run results)
# ---------------------------------------------------------------------------


def table_path(results_root: Path) -> Path:
    # Lives at the base of the autoresearch results tree.
    return results_root / "autoresearch" / "experiments.csv"


def _config_rel(cfg: "Any", results_root: Path) -> str:
    """The treatment config path relative to the testbed root (= results_root's
    parent) — the row's identity in the global table."""
    if cfg.config_path:
        try:
            return os.path.relpath(cfg.config_path, results_root.parent)
        except ValueError:
            return str(cfg.config_path)
    iface = cfg.interfaces[0] if cfg.interfaces else None
    return _config_rel_path(
        iface.platform if iface else "", cfg.research_question or 0,
        cfg.treatment, iface.interface if iface else "", cfg.optimization_variable or "",
    )


def _design_cells(cfg: "Any", config_rel: str) -> dict:
    iface = cfg.interfaces[0] if cfg.interfaces else None
    goals = [(g.metric, g.direction) for g in cfg.goals]
    return {
        "experiment": cfg.experiment,
        "research_question": cfg.research_question if cfg.research_question is not None else "",
        "treatment": cfg.treatment,
        "config": config_rel,
        "researcher_model": cfg.researcher_model,
        "engineer_model": cfg.engineer_model,
        "platform": iface.platform if iface else "",
        "interface": iface.interface if iface else "",
        "language": cfg.language or "",
        "skills": _skills_label(cfg.improve or [], cfg.skills),
        "optimization_variable": cfg.optimization_variable or "",
        "goals": _goals_str(goals),
        "time": cfg.time if cfg.time is not None else "",
        "budget": _budget_arr(cfg.budget) if cfg.budget else "",
    }


def read_table(results_root: Path) -> list[dict]:
    path = table_path(results_root)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_table(results_root: Path, rows: list[dict]) -> Path:
    path = table_path(results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in EXPERIMENT_FIELDS})
    return path


def annotate_version(results_root: Path, config_rel: str, version: str,
                     updates: dict, interface: str | None = None) -> int:
    """Fill the per-version annotation columns (hypothesis/change/verdict/…) on
    every row of `(config_rel, version)` — optionally scoped to one `interface`
    leaf. The researcher calls `banter annotate-version` once per finished
    version; values land on all that version's challenge rows. Returns the count
    of rows updated. `version` matches the stored form (e.g. "v3")."""
    path = table_path(results_root)
    if not path.exists():
        return 0
    rows = read_table(results_root)
    n = 0
    for r in rows:
        if (r.get("config") == config_rel
                and str(r.get("version", "")) == str(version)
                and (interface is None or r.get("interface") == interface)):
            for k, v in updates.items():
                if k in ANNOTATION_FIELDS and v is not None:
                    r[k] = str(v)
            n += 1
    if n:
        _write_table(results_root, rows)
    return n


def clear_treatment(results_root: Path, cfg: "Any") -> None:
    """Drop a treatment's existing rows from the global table (called when a
    treatment is (re)run, so its prior rows don't linger)."""
    path = table_path(results_root)
    if not path.exists():
        return
    config_rel = _config_rel(cfg, results_root)
    kept = [r for r in read_table(results_root) if r.get("config") != config_rel]
    _write_table(results_root, kept)


def append_run(results_root: Path, cfg: "Any", result_row: dict) -> Path:
    """Append ONE exploded row (`result_row` = a results.Row as a dict) for this
    treatment to the global `experiments.csv`, tagged with the config's design
    metadata. The runner calls this directly per (version, task, challenge) —
    there is no per-run results.csv. Creates the file with a header if absent.
    """
    config_rel = _config_rel(cfg, results_root)
    row = {k: "" for k in EXPERIMENT_FIELDS}
    row.update(_design_cells(cfg, config_rel))
    # platform/interface come from the ACTUAL run (a config may span several
    # interfaces, each its own leaf), overriding the design default.
    if result_row.get("platform"):
        row["platform"] = result_row["platform"]
    if result_row.get("interface"):
        row["interface"] = result_row["interface"]
    # Goals are interface-specific for delegation metrics: keep only THIS
    # interface's `*_calls` goal (you can't maximize cli+mcp+sdk at once).
    leaf_goals = goals_for_interface(
        [(g.metric, g.direction) for g in cfg.goals], row["interface"])
    row["goals"] = _goals_str(leaf_goals)
    # optimization_variable is derived from how many goals this leaf optimizes:
    # 1 → univariate, 2 → bivariate, 3 → multivariate.
    row["optimization_variable"] = _optvar_for_count(len(leaf_goals))
    row["version"] = result_row.get("version", "")
    row["task"] = result_row.get("task", "")
    row["challenge"] = result_row.get("challenge", "")
    row["started_at"] = result_row.get("started_at", "")
    for col in RESULT_FIELDS:
        row[col] = result_row.get(col, "")
    # `total_input_tokens`/`total_output_tokens` aren't on the engineer Row;
    # before researcher attribution the total equals the engineer side.
    if row.get("total_input_tokens", "") == "":
        row["total_input_tokens"] = result_row.get("eng_input_tokens", "")
    if row.get("total_output_tokens", "") == "":
        row["total_output_tokens"] = result_row.get("eng_output_tokens", "")

    path = table_path(results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in EXPERIMENT_FIELDS})

    # Regenerate AND execute the global analysis notebook after every appended
    # row so its diagrams propagate live as the experiment runs. Executing spawns
    # a Jupyter kernel + re-runs all cells (a few seconds), so it adds overhead
    # per engineer run — acceptable since each engineer run is minutes long, and
    # live charts are the point. Best-effort + non-fatal (a kernel/exec failure
    # must never break the run; the session-end refresh re-executes regardless).
    try:
        build_global_notebook(results_root.parent, execute=True)
    except Exception:
        pass
    return path


def attribute_researcher(
    results_root: Path,
    cfg: "Any",
    platform: str,
    interface: str,
    researcher_usage: dict,
) -> Path | None:
    """Distribute the researcher's tokens + wall time + cost across THIS leaf's
    rows in the global table (the researcher is shared overhead), and set every
    `total_* = eng_* + res_*`. Called once at session end. `researcher_usage`
    has keys `input_tokens`, `output_tokens`, `total_tokens`, `wall_s`,
    `cost_usd`. Idempotent — totals are recomputed from the stored engineer
    columns each call.
    """
    path = table_path(results_root)
    if not path.exists():
        return None
    config_rel = _config_rel(cfg, results_root)
    rows = read_table(results_root)

    def is_leaf(r: dict) -> bool:
        return (r.get("config") == config_rel
                and r.get("platform") == platform
                and r.get("interface") == interface)

    # DEAD rows (no valid submission) were written with ALL metrics zeroed and
    # an `error` set — they must STAY zeroed. Exclude them from researcher
    # attribution so the shared overhead lands only on rows that produced a
    # result, and the dead row's `total_*` isn't resurrected above zero.
    def is_dead(r: dict) -> bool:
        return bool((r.get("error") or "").strip())

    leaf = [r for r in rows if is_leaf(r) and not is_dead(r)]
    n = len(leaf) or 1
    for r in leaf:
        # Per-row researcher share (equal split across the leaf's rows).
        for res_col, usage_key in _RES_FROM.items():
            share = (_fnum(researcher_usage.get(usage_key)) or 0.0) / n
            r[res_col] = f"{share:.6f}"
        # total = engineer + researcher, for every paired metric.
        for eng_col, res_col, total_col in _ATTR_PAIRS:
            eng = _fnum(r.get(eng_col)) or 0.0
            res = _fnum(r.get(res_col)) or 0.0
            r[total_col] = f"{eng + res:.6f}"
    _write_table(results_root, rows)
    return path


def write_header_only(results_root: Path) -> Path:
    path = table_path(results_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDS).writeheader()
    return path


# ---------------------------------------------------------------------------
# Global analysis notebook (does the math)
# ---------------------------------------------------------------------------

_NOTEBOOK_SETUP = '''\
import json
from pathlib import Path
import pandas as pd
from banter import experiments as ex

# Locate experiments.csv robustly so this notebook runs from any cwd (next to
# the notebook, or from the testbed root). Empty frame if there are no runs yet.
_candidates = [Path("experiments.csv"), Path("results/autoresearch/experiments.csv")]
_csv = next((p for p in _candidates if p.exists()), None)
df = (pd.read_csv(_csv, dtype=str, keep_default_na=False)  # strings → blanks stay ''
      if _csv is not None else pd.DataFrame(columns=ex.EXPERIMENT_FIELDS))
print(f"loaded {len(df)} row(s)" + (f" from {_csv}" if _csv is not None else " (no experiments.csv yet)"))
df
'''

_NOTEBOOK_DETAIL = '''\
# Full granularity: every (version, task, challenge) with its key metrics —
# one row per engineer run, sorted by treatment then version then challenge.
_cols = [c for c in [
    "config", "interface", "version", "task", "challenge",
    "score", "total_tokens", "python_calls",
    "cli_calls", "mcp_calls", "sdk_calls",
    "eng_total_tokens", "res_total_tokens",
    "total_wall_time_s", "total_cost",
] if c in df.columns]
if not df.empty and _cols:
    detail = df[_cols].copy()
    if "version" in detail.columns:
        detail["_v"] = detail["version"].str.lstrip("v").replace("", "-1").astype(int)
        detail = detail.sort_values(["config", "_v", "task", "challenge"]).drop(columns="_v")
    display(detail)
else:
    print("no runs yet")
'''

_NOTEBOOK_BEST = '''\
# Per-treatment: best version by the normalized composite over its goals, with
# baseline (v0) vs best for every metric. All math lives here, not in the CSV.
# Group by (config, interface): a config may run several interfaces, and each
# interface is its own leaf with its own goals (you can't mix cli/mcp/sdk).
runs = df[df["version"] != ""] if "version" in df.columns else df.iloc[0:0]
summary = []
for (config, interface), g in runs.groupby(["config", "interface"]):
    goals = ex.parse_goals_str(g["goals"].iloc[0])
    per = ex.aggregate_by_version(g.to_dict("records"), ex.METRICS)
    if not per:
        continue
    base, best = min(per), ex.best_version(per, goals)
    rec = {
        "treatment": g["treatment"].iloc[0],
        "optimization_variable": g["optimization_variable"].iloc[0],
        "interface": interface,
        "config": config,
        "n_versions": len(per),
        "best_version": f"v{best}",
    }
    for m in ex.METRICS:
        rec[f"{m}_baseline"] = per[base].get(m)
        rec[f"{m}_best"] = per[best].get(m)
    summary.append(rec)
summary_df = (pd.DataFrame(summary).sort_values(["treatment", "interface"])
              if summary else pd.DataFrame())
summary_df
'''

_NOTEBOOK_CHART = '''\
# ONE LINE GRAPH PER TREATMENT: how that treatment's goal metrics develop across
# versions (x = v0..vN). Each point is the mean across the treatment's
# tasks/challenges at that version. Goals are normalized to [0, 1] on the y-axis
# (1 = the best version for that goal, in its optimize direction) so goals on
# different scales (score, python_calls, total_tokens, …) are comparable on one
# axis — raw values are in the detail table above. A dashed line shows the
# overall composite. Treatments are keyed by the `treatment` id; rows with no
# treatment id are grouped by their shared parameters (config / interface /
# optimization_variable / skills).
import matplotlib.pyplot as plt

runs = df[df["version"] != ""].copy() if "version" in df.columns else df.iloc[0:0].copy()

def _treatment_key(r):
    t = str(r.get("treatment", "")).strip()
    if t:
        return f"exp{r.get('experiment','')}-t{t}"
    return "|".join(str(r.get(c, "")) for c in
                    ("config", "interface", "optimization_variable", "skills"))

def _treatment_title(g):
    r = g.iloc[0]
    t = str(r.get("treatment", "")).strip()
    head = f"Treatment {t}" if t else "Treatment (grouped by parameters)"
    bits = [b for b in (
        f"RQ{r.get('research_question','')}" if str(r.get("research_question", "")).strip() else "",
        str(r.get("interface", "")),
        str(r.get("optimization_variable", "")),
        f"skills={r.get('skills','')}" if str(r.get("skills", "")).strip() else "",
    ) if b]
    return head + ("  [" + " · ".join(bits) + "]" if bits else "")

if runs.empty:
    print("no runs yet — nothing to chart")
else:
    runs["_tkey"] = runs.apply(_treatment_key, axis=1)
    for _tk, g in runs.groupby("_tkey"):
        goals = ex.parse_goals_str(g["goals"].iloc[0])
        per = ex.aggregate_by_version(g.to_dict("records"), ex.METRICS)
        if not per or not goals:
            continue
        bd = ex.composite_breakdown(per, goals)
        vs = sorted(per)
        fig, ax = plt.subplots(figsize=(7, 4))
        for metric, direction in goals:
            ys = [bd[v]["contributions"].get(metric) for v in vs]
            ax.plot(vs, ys, marker="o",
                    label=f"{metric} ({'min' if direction == 'minimize' else 'max'})")
        if len(goals) > 1:   # overall composite (scaled to [0,1])
            ax.plot(vs, [bd[v]["composite"] / len(goals) for v in vs],
                    "k--", lw=2, label="composite")
        ax.set_title(_treatment_title(g))
        ax.set_xlabel("version"); ax.set_ylabel("normalized (1 = best)")
        ax.set_xticks(vs); ax.set_xticklabels([f"v{v}" for v in vs])
        ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        plt.tight_layout(); plt.show()
'''


def build_global_notebook(testbed_root: Path, execute: bool = False) -> Path:
    """Write `results/autoresearch/analysis.ipynb` — the single global analysis,
    sitting next to `experiments.csv`, that computes best version /
    baseline-vs-best / charts. Always runnable: the cells tolerate a missing or
    empty table, so it executes cleanly at any time (before or after any runs).

    With `execute=True`, the notebook is run in place (kernel cwd = its own dir)
    and the outputs are embedded. Raises if execution fails (so callers can
    report it) — writing the (unexecuted) notebook still happens first.
    """
    from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
    import nbformat

    results_root = (testbed_root / "results" / "autoresearch")
    results_root.mkdir(parents=True, exist_ok=True)
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# Experiments — global analysis\n\n"
            "Aggregates every treatment in `experiments.csv` (the factual, exploded "
            "global table — one row per version × task × challenge). All derived "
            "math lives here: the normalized composite picks each treatment's best "
            "version, shown against its v0 baseline.\n\n"
            "Runnable at any time (handles an empty/missing table). Regenerate + "
            "execute with `banter experiments refresh`."
        ),
        new_markdown_cell("## Load"),
        new_code_cell(_NOTEBOOK_SETUP),
        new_markdown_cell("## Every (version × task × challenge)"),
        new_code_cell(_NOTEBOOK_DETAIL),
        new_markdown_cell("## Goal development across versions — one graph per treatment"),
        new_code_cell(_NOTEBOOK_CHART),
        new_markdown_cell("## Best version per treatment (normalized composite)"),
        new_code_cell(_NOTEBOOK_BEST),
    ]
    path = results_root / "analysis.ipynb"
    # Write first so the notebook file exists even if execution is skipped/fails.
    with path.open("w") as f:
        nbformat.write(nb, f)
    if execute:
        import sys
        from nbclient import NotebookClient
        from jupyter_client.manager import KernelManager
        km = KernelManager()
        km.kernel_cmd = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        try:
            NotebookClient(nb, km=km, timeout=180,
                           resources={"metadata": {"path": str(results_root)}}).execute()
            with path.open("w") as f:   # re-write with embedded outputs
                nbformat.write(nb, f)
        finally:
            # Explicitly shut the kernel down, else ipykernel lingers and prints
            # "Parent appears to have exited" after the caller returns.
            try:
                if km.is_alive():
                    km.shutdown_kernel(now=True)
            except Exception:
                pass
    return path
