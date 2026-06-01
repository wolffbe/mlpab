"""Generate an `analysis.ipynb` for each autoresearch run.

Reads `<run>/results.csv` (raw per-challenge rows, written by the runner)
and produces a self-contained notebook with one line chart per goal metric,
showing how that metric develops across versions. The notebook is pre-
executed at generation time so the charts are embedded in the `.ipynb` —
opening it in Jupyter (or on GitHub) renders the plots without re-running
anything. The notebook itself computes per-version means from the raw rows.

Triggered at end-of-run (`autoresearch.run_autoresearch`). Pure data →
notebook transform; no AI involved.

Chart semantics per goal (`{metric, direction}`):
  * x = version index (parsed from the `version` column, e.g. "v2" → 2)
  * y = the metric value (mean across this version's challenges)
  * Each chart title carries the direction (maximize / minimize) so the
    desired slope is obvious at a glance.

Notebook execution is best-effort: if execution fails we still write the
unexecuted notebook so the user can run the cells by hand.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


_VERSION_RE = re.compile(r"v(\d+)$")


def _version_index(name: str) -> int | None:
    """Extract N from `v<N>`; returns None for non-version folder names."""
    m = _VERSION_RE.match(name)
    return int(m.group(1)) if m else None


def _setup_cell(master_csv_relpath: str, run_id: str) -> Any:
    """Code cell that loads the run's results.csv and computes per-version
    means for every numeric column.

    `version` values are "v0", "v1", … — stripped to integer for the x-axis.
    The notebook filters `run == <run_id>` defensively in case the CSV was
    concatenated with rows from other runs.
    """
    src = (
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        f"master = pd.read_csv({json.dumps(master_csv_relpath)})\n"
        f"# Filter to this run ({json.dumps(run_id)}).\n"
        f"raw = master[master['run'].astype(str) == {json.dumps(run_id)}].copy()\n"
        "raw['version_idx'] = raw['version'].astype(str).str.extract(r'v?(\\d+)').astype(int)\n"
        "# Average every numeric column per version; non-numeric cols are dropped.\n"
        "numeric = raw.select_dtypes(include='number')\n"
        "df = numeric.groupby('version_idx', as_index=False).mean()\n"
        "# `*_avg` columns are a cumulative running mean across rows in append\n"
        "# order, so the value 'as of' a version is its LAST row — NOT the mean of\n"
        "# the running means within the version. Overwrite those columns with the\n"
        "# per-version last value so the dotted overlay is the true running average.\n"
        "_avg_cols = [c for c in numeric.columns if c.endswith('_avg') or c.endswith('_avg_s')]\n"
        "if _avg_cols:\n"
        "    _last = numeric.groupby('version_idx', as_index=False)[_avg_cols].last()\n"
        "    df = df.drop(columns=_avg_cols).merge(_last, on='version_idx')\n"
        "df = df.sort_values('version_idx').reset_index(drop=True)\n"
        "df['n_challenges'] = raw.groupby('version_idx').size().reindex(df['version_idx']).values\n"
        "df\n"
    )
    return new_code_cell(src)


def _plot_cell(metric: str, direction: str, optimized: bool, avg_col: str | None = None) -> Any:
    """Code cell rendering one metric as a single line chart over versions.

    The solid line is the mean of `metric` across all tasks/challenges in each
    version (already averaged in `results.csv`); `n_challenges` is annotated
    above each point. When `avg_col` is set, a second dotted line shows the
    cumulative average across all runs (read straight from `results.csv`), so
    convergence is visible as the two lines flatten and meet. Optimized goals
    get a thicker line + a `(optimized)` title.
    """
    src = (
        f"metric = {json.dumps(metric)}\n"
        f"direction = {json.dumps(direction)}\n"
        f"optimized = {repr(bool(optimized))}\n"
        f"avg_col = {repr(avg_col)}\n"
        "if metric not in df.columns:\n"
        "    print(f'(metric {metric!r} not in results.csv — skipping)')\n"
        "elif pd.to_numeric(df[metric], errors='coerce').notna().sum() == 0:\n"
        "    print(f'(no data for {metric!r} in this run — skipping)')\n"
        "else:\n"
        "    fig, ax = plt.subplots(figsize=(7, 4))\n"
        "    y = pd.to_numeric(df[metric], errors='coerce')\n"
        "    lw = 2.5 if optimized else 1.5\n"
        "    alpha = 1.0 if optimized else 0.85\n"
        "    ax.plot(df['version_idx'], y, marker='o', linewidth=lw, alpha=alpha, label=metric)\n"
        "    # Overlay the cumulative rolling average as a dotted line.\n"
        "    if avg_col and avg_col in df.columns:\n"
        "        ya = pd.to_numeric(df[avg_col], errors='coerce')\n"
        "        if ya.notna().any():\n"
        "            ax.plot(df['version_idx'], ya, linestyle=':', linewidth=2,\n"
        "                    color='gray', label='rolling avg')\n"
        "            ax.legend(fontsize=8)\n"
        "    # Annotate sample size (above) when more than one challenge per version.\n"
        "    if 'n_challenges' in df.columns and df['n_challenges'].max() > 1:\n"
        "        for xi, yi, ni in zip(df['version_idx'], y, df['n_challenges']):\n"
        "            if pd.notna(yi):\n"
        "                ax.annotate(f'n={ni}', (xi, yi), textcoords='offset points',\n"
        "                            xytext=(0, 8), ha='center', fontsize=7, alpha=0.7)\n"
        "    ax.set_xlabel('version')\n"
        "    ax.set_ylabel(f'mean {metric}')\n"
        "    suffix = ' (optimized)' if optimized else ''\n"
        "    ax.set_title(f'{direction}({metric}) over versions{suffix}')\n"
        "    ax.grid(True, alpha=0.3)\n"
        "    ax.set_xticks(sorted(df['version_idx'].unique()))\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
    )
    return new_code_cell(src)


def _intro_cell(
    run_id: str,
    tracked: list[tuple[str, str, bool]],
    n_versions: int,
) -> Any:
    """Markdown header listing the canonical tracked metrics + which are
    flagged as optimization goals in the config (the rest are tracked but
    not actively optimized)."""
    lines = []
    for metric, direction, optimized in tracked:
        marker = " — **optimized**" if optimized else ""
        lines.append(f"- **{direction}**(`{metric}`){marker}")
    metrics_md = "\n".join(lines)
    return new_markdown_cell(
        f"# Autoresearch run `{run_id}` — analysis\n\n"
        f"Generated from `results.csv` (one row per task × challenge × "
        f"version — the notebook computes per-version means itself). "
        f"This run produced **{n_versions} version(s)**.\n\n"
        f"## Tracked metrics\n{metrics_md}\n\n"
        "Every metric is charted across versions as a solid line, with a second "
        "**dotted line** showing the cumulative average across all runs so "
        "convergence is visible as the two lines flatten and meet. The "
        "**optimized** marker "
        "indicates which ones the researcher is actively driving per the "
        "config's `goals` block; the rest are tracked for situational "
        "awareness (e.g. an interface refactor that improves `score` while "
        "ballooning `wall_time_s`)."
    )


def _resolve_tracked(
    goals: list[tuple[str, str]],
) -> list[tuple[str, str, bool]]:
    """Merge config goals with the canonical `TRACKED_METRICS` list.

    Returns `[(metric, direction, optimized)]` where:
      * Every entry in `TRACKED_METRICS` is included (always charted), in its
        declared order. If the config redefines its direction, the config wins.
      * Any extra config goals (metrics not in the canonical list) are
        appended at the end so non-default optimization targets are charted too.
      * `optimized=True` iff the metric appears in the config goals list.
    """
    from banter import results as results_mod
    goals_by_metric = dict(goals)
    out: list[tuple[str, str, bool]] = []
    seen: set[str] = set()
    for metric, default_dir in results_mod.TRACKED_METRICS:
        direction = goals_by_metric.get(metric, default_dir)
        out.append((metric, direction, metric in goals_by_metric))
        seen.add(metric)
    for metric, direction in goals:
        if metric not in seen:
            out.append((metric, direction, True))
            seen.add(metric)
    return out


def _benchmark_chart_cell(metric: str, direction: str) -> Any:
    """Bar chart cell: one bar per (task, challenge), labelled.

    Benchmark has no versions (a single session of independent challenges), so
    the x-axis is the `(task, challenge)` pair (or just challenge when there's
    one task). No rolling-average overlay — there's nothing to accumulate across.
    """
    src = (
        f"metric = {json.dumps(metric)}\n"
        f"direction = {json.dumps(direction)}\n"
        "work = (raw.assign(_y=pd.to_numeric(raw[metric], errors='coerce')).dropna(subset=['_y'])\n"
        "        if metric in raw.columns else raw.iloc[0:0].assign(_y=[]))\n"
        "if work.empty:\n"
        "    print(f'(no data for {metric!r} in this run — skipping)')\n"
        "else:\n"
        "    work = work.sort_values(['task', 'challenge']).reset_index(drop=True)\n"
        "    labels = [f'{t}/{c}' if t and t != 'no_task' else c\n"
        "              for t, c in zip(work['task'].astype(str), work['challenge'].astype(str))]\n"
        "    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(work)), 4))\n"
        "    ax.bar(range(len(work)), work['_y'])\n"
        "    ax.set_xticks(range(len(work)))\n"
        "    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)\n"
        "    ax.set_ylabel(metric)\n"
        "    ax.set_title(f'{direction}({metric}) per (task, challenge)')\n"
        "    ax.grid(True, axis='y', alpha=0.3)\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
    )
    return new_code_cell(src)


def build_benchmark_notebook(
    run_dir: Path,
    run_id: str,
    execute: bool = True,
    results_csv: Path | None = None,
) -> Path | None:
    """Generate `<run_dir>/analysis.ipynb` for a benchmark session.

    Charts every metric in `results.TRACKED_METRICS` as a bar over the
    session's (task, challenge) pairs. There are no versions to plot a
    line over, so each metric becomes one labelled bar chart.
    """
    if results_csv is None:
        results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return None

    # Make sure there are rows beyond the header.
    import csv as _csv
    with open(results_csv, newline="") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        return None

    import os
    csv_rel = os.path.relpath(results_csv, run_dir)
    from banter import results as results_mod
    tracked = results_mod.BENCHMARK_TRACKED_METRICS

    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            f"# Benchmark run `{run_id}` — analysis\n\n"
            f"Generated from `results.csv` (one row per (task, challenge)). "
            f"This run produced **{len(rows)} run(s)**.\n\n"
            "## Tracked metrics\n"
            + "\n".join(f"- **{d}**(`{m}`)" for m, d in tracked) + "\n\n"
            "Each metric is shown as a bar over every `(task, challenge)` pair."
        ),
        new_markdown_cell("## Setup"),
        new_code_cell(
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            f"raw = pd.read_csv({json.dumps(csv_rel)})\n"
            "raw\n"
        ),
        new_markdown_cell("## Metrics per (task, challenge)"),
    ]
    for metric, direction in tracked:
        nb.cells.append(new_markdown_cell(f"### `{direction}({metric})`"))
        nb.cells.append(_benchmark_chart_cell(metric, direction))

    if execute:
        import sys
        from nbclient import NotebookClient
        from jupyter_client.manager import KernelManager
        km = KernelManager()
        km.kernel_cmd = [
            sys.executable, "-m", "ipykernel_launcher",
            "-f", "{connection_file}",
        ]
        client = NotebookClient(
            nb, km=km, timeout=120,
            resources={"metadata": {"path": str(run_dir)}},
        )
        try:
            client.execute()
        except Exception as e:
            print(f"[notebook] pre-execution failed ({type(e).__name__}: {e}); "
                  f"writing unexecuted notebook.", flush=True)
        finally:
            try:
                if km.is_alive():
                    km.shutdown_kernel(now=True)
            except Exception:
                pass

    out_path = run_dir / "analysis.ipynb"
    with open(out_path, "w") as f:
        nbformat.write(nb, f)
    return out_path


def build_run_notebook(
    run_dir: Path,
    goals: list[tuple[str, str]],
    run_id: str,
    execute: bool = True,
    master_csv: Path | None = None,
) -> Path | None:
    """Generate `<run_dir>/analysis.ipynb` from `<run_dir>/results.csv`.

    The run's results.csv lives in the same dir as the notebook (one CSV
    per autoresearch run, accumulating rows across every version). The
    notebook filters `run == run_id` defensively and then charts
    every metric in `results.TRACKED_METRICS` plus any extra
    `(metric, direction)` from the config goals. Metrics in `goals` are
    marked `(optimized)` in the chart titles.

    Returns the path to the written notebook, or None if there are no
    version rows for `run_id`.
    """
    if master_csv is None:
        master_csv = run_dir / "results.csv"
    if not master_csv.exists():
        return None

    # Count this run's versions (best-effort; pandas not imported at module
    # top because the notebook itself is the consumer).
    import csv as _csv
    incrs: set[str] = set()
    with open(master_csv, newline="") as f:
        for row in _csv.DictReader(f):
            if str(row.get("run", "")) == str(run_id) and row.get("version"):
                incrs.add(row["version"])
    if not incrs:
        return None

    # Path inside the notebook is just `results.csv` (same dir).
    import os
    master_rel = os.path.relpath(master_csv, run_dir)

    from banter import results as results_mod
    tracked = _resolve_tracked(goals)
    nb = new_notebook()
    nb.cells = [
        _intro_cell(run_id, tracked, n_versions=len(incrs)),
        new_markdown_cell("## Setup"),
        _setup_cell(master_rel, run_id),
        new_markdown_cell("## Metrics per version"),
    ]
    for metric, direction, optimized in tracked:
        marker = " — optimized" if optimized else ""
        nb.cells.append(new_markdown_cell(f"### `{direction}({metric})`{marker}"))
        nb.cells.append(_plot_cell(metric, direction, optimized, results_mod.rolling_avg_col(metric)))

    if execute:
        import sys
        from nbclient import NotebookClient
        from jupyter_client.manager import KernelManager
        # Drive the kernel from THIS venv's interpreter so the notebook picks
        # up matplotlib + pandas from the same environment that generated it.
        km = KernelManager()
        km.kernel_cmd = [
            sys.executable, "-m", "ipykernel_launcher",
            "-f", "{connection_file}",
        ]
        client = NotebookClient(
            nb, km=km, timeout=120,
            resources={"metadata": {"path": str(run_dir)}},
        )
        try:
            client.execute()
        except Exception as e:
            # Best effort: write the unexecuted notebook anyway so the user
            # can still open + run it by hand.
            print(f"[notebook] pre-execution failed ({type(e).__name__}: {e}); "
                  f"writing unexecuted notebook.", flush=True)
        finally:
            # CRITICAL: explicitly shut down the kernel subprocess. Without
            # this, the ipykernel runs orphaned after banter exits and prints
            # `[IPKernelApp] WARNING | Parent appears to have exited, shutting
            # down.` to the terminal seconds AFTER your shell prompt returned.
            try:
                if km.is_alive():
                    km.shutdown_kernel(now=True)
            except Exception:
                # If the kernel never came up (e.g. ipykernel missing) the
                # shutdown call itself can fail — non-fatal.
                pass

    out_path = run_dir / "analysis.ipynb"
    with open(out_path, "w") as f:
        nbformat.write(nb, f)
    return out_path
