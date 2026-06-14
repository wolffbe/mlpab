"""Generate the GLOBAL `results.ipynb`.

Reads the global results table (results/results.csv — one row per
execution across ALL configs) and produces a single self-contained notebook at
the results root:

  1. the raw execution list,
  2. an aggregated table — one row per (config, model, platform, interface,
     version, skills), metrics averaged across tasks and repeats (`n`), plus a
     final `(all)` average row,
  3. one section per config (headline), one subsection per model, and within
     it one bar chart per tracked metric — one bar per (platform, interface,
     version, skills), averaged across tasks/repeats, with a dashed line at
     that config/model's overall average.

Triggered after every treatment run (`treatments.run_treatments`). Pure data →
notebook transform; no AI involved.

Notebook execution is best-effort: if execution fails we still write the
unexecuted notebook so the user can run the cells by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

# Bars within a chart: every distinct interface build under a (config, model).
BAR_FIELDS = ["platform", "interface", "version", "skills"]
# Aggregated-table grouping: the bar identity plus the section identity.
GROUP_FIELDS = ["config", "model"] + BAR_FIELDS


def _setup_cell(csv_rel: str) -> Any:
    return new_code_cell(
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        f"raw = pd.read_csv({json.dumps(csv_rel)})\n"
        "raw\n"
    )


def _aggregate_cell(metrics: list[str]) -> Any:
    """One row per GROUP_FIELDS combo: metrics averaged across tasks and
    repeats; `runs` counts the executions averaged; the final `(all)` row
    averages over every execution."""
    src = (
        f"GROUP = {json.dumps(GROUP_FIELDS)}\n"
        f"METRICS = [m for m in {json.dumps(metrics)} if m in raw.columns]\n"
        "num = raw.copy()\n"
        "for m in METRICS:\n"
        "    num[m] = pd.to_numeric(num[m], errors='coerce')\n"
        "agg = (num.groupby(GROUP, dropna=False)\n"
        "          .agg(runs=('task', 'size'), **{m: (m, 'mean') for m in METRICS})\n"
        "          .reset_index())\n"
        "overall = {c: '(all)' for c in GROUP}\n"
        "overall['runs'] = len(num)\n"
        "overall.update({m: num[m].mean() for m in METRICS})\n"
        "pd.concat([agg, pd.DataFrame([overall])], ignore_index=True)\n"
    )
    return new_code_cell(src)


def _results_chart_cell(config: str, model: str, metric: str) -> Any:
    """Bar chart cell for one (config, model, metric): one bar per
    (platform, interface, version, skills), y = mean across tasks and repeats;
    dashed line = the config/model average across all its rows."""
    src = (
        f"config, model = {json.dumps(config)}, {json.dumps(model)}\n"
        f"metric = {json.dumps(metric)}\n"
        "sub = num[(num['config'].astype(str) == config) & (num['model'].astype(str) == model)]\n"
        "work = (sub.dropna(subset=[metric])\n"
        f"           .groupby({json.dumps(BAR_FIELDS)}, dropna=False)[metric]\n"
        "           .mean().reset_index()) if metric in sub.columns else sub.iloc[0:0]\n"
        "if work.empty:\n"
        "    print(f'(no data for {metric!r} in {config}/{model} — skipping)')\n"
        "else:\n"
        f"    labels = ['/'.join(str(r[c]) for c in {json.dumps(BAR_FIELDS)})\n"
        "              for _, r in work.iterrows()]\n"
        "    avg = sub[metric].mean()\n"
        "    fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(work)), 4))\n"
        "    ax.bar(range(len(work)), work[metric])\n"
        "    ax.axhline(avg, color='tab:red', linestyle='--', linewidth=1,\n"
        "               label=f'average = {avg:.4g}')\n"
        "    ax.set_xticks(range(len(work)))\n"
        "    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)\n"
        "    ax.set_ylabel(metric)\n"
        "    ax.set_title(f'{metric} — mean across tasks/repeats')\n"
        "    ax.grid(True, axis='y', alpha=0.3)\n"
        "    ax.legend()\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
    )
    return new_code_cell(src)


def build_results_notebook(
    results_root: Path,
    execute: bool = True,
    results_csv: Path | None = None,
) -> Path | None:
    """Generate the global `<results_root>/results.ipynb` from the global
    table `<results_root>/results.csv` (one row per execution, all configs)."""
    if results_csv is None:
        results_csv = results_root / "results.csv"
    if not results_csv.exists():
        return None

    # Make sure there are rows beyond the header.
    import csv as _csv

    with open(results_csv, newline="") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        return None

    import os

    csv_rel = os.path.relpath(results_csv, results_root)
    from mlpab import results as results_mod

    tracked = results_mod.RESULTS_TRACKED_METRICS

    # Section structure mirrors the data: config → model → metric charts.
    sections: dict[str, list[str]] = {}
    for r in rows:
        models = sections.setdefault(str(r.get("config", "")), [])
        model = str(r.get("model", ""))
        if model not in models:
            models.append(model)

    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# Results — global analysis\n\n"
            f"Generated from `results.csv` — one row per execution across ALL "
            f"configs (**{len(rows)} execution(s)**).\n\n"
            "One section per config, one subsection per model, one chart per "
            "tracked metric: one bar per `(platform, interface, version, "
            "skills)`, averaged across tasks and repeats (`n`); "
            "the dashed line is that config/model's overall average.\n\n"
            "## Tracked metrics\n" + "\n".join(f"- `{m}`" for m in tracked)
        ),
        new_markdown_cell("## Raw executions"),
        _setup_cell(csv_rel),
        new_markdown_cell("## Averages per (config, model, platform, interface, version, skills)"),
        _aggregate_cell(list(tracked)),
    ]
    for config in sorted(sections):
        nb.cells.append(new_markdown_cell(f"## Config `{config}`"))
        for model in sorted(sections[config]):
            nb.cells.append(new_markdown_cell(f"### Model `{model}`"))
            for metric in tracked:
                nb.cells.append(new_markdown_cell(f"#### `{metric}`"))
                nb.cells.append(_results_chart_cell(config, model, metric))

    if execute:
        import sys

        from jupyter_client.manager import KernelManager
        from nbclient import NotebookClient

        km = KernelManager()
        km.kernel_cmd = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        client = NotebookClient(
            nb,
            km=km,
            timeout=120,
            resources={"metadata": {"path": str(results_root)}},
        )
        try:
            client.execute()
        except Exception as e:
            print(
                f"[mlpab] pre-execution failed ({type(e).__name__}: {e}); "
                f"writing unexecuted notebook.",
                flush=True,
            )
        finally:
            try:
                if km.is_alive():
                    km.shutdown_kernel(now=True)
            except Exception:
                pass

    # Atomic replace: parallel treatment processes refresh this notebook after
    # every run; an in-place write could leave torn JSON for a concurrent reader.
    import os
    import tempfile

    out_path = results_root / "results.ipynb"
    fd, tmp_name = tempfile.mkstemp(dir=str(results_root), prefix=".results.ipynb.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            nbformat.write(nb, f)
        os.replace(tmp_name, out_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return out_path
