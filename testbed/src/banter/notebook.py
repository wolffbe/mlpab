"""Generate the per-benchmark-session `analysis.ipynb`.

Reads a benchmark run's `results.csv` (one row per (task, challenge)) and
produces a self-contained notebook with one labelled bar chart per tracked
metric. The notebook is pre-executed at generation time so the charts are
embedded in the `.ipynb` — opening it in Jupyter (or on GitHub) renders the
plots without re-running anything.

Triggered at end-of-benchmark (`benchmark.run_benchmark`). Pure data → notebook
transform; no AI involved. (Autoresearch uses the single GLOBAL notebook built
by `experiments.build_global_notebook`, not this module.)

Notebook execution is best-effort: if execution fails we still write the
unexecuted notebook so the user can run the cells by hand.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


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
