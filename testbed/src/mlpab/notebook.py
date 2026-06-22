"""Generate the GLOBAL `results.ipynb`.

Reads the global results table (results/results.csv — one row per execution
across ALL configs) and produces a single self-contained notebook at the
results root with four headline diagrams — one per metric that counts — each a
grouped bar chart comparing every config across its interface/skills combos:

  1. **pass rate** (`asserts_passed / total_asserts`) — higher is better,
  2. **local_time_s** — lower is better,
  3. **cost_usd** — lower is better,
  4. **llm_calls** — lower is better.

Bars are grouped by config; within each config there are four bars for the
interface (cli/sdk) × skills (with/without) combos, each averaged across all
matching executions (every task, version, repeat). The dashed line is the mean
across all configs.

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

# The four headline metrics, in display order. `col` is the column in `num`
# (`pass_rate` is derived in the setup cell); `direction` drives colour and the
# ↑/↓ marker; higher pass rate is better, lower time/cost/calls is better.
KEY_METRICS = [
    {"col": "pass_rate", "label": "pass rate (asserts_passed / total_asserts)", "direction": "max"},
    {"col": "local_time_s", "label": "local_time_s", "direction": "min"},
    {"col": "cost_usd", "label": "cost_usd", "direction": "min"},
    {"col": "llm_calls", "label": "llm_calls", "direction": "min"},
]


def _setup_cell(csv_rel: str) -> Any:
    """Load the table, coerce the needed columns to numeric, derive `pass_rate`,
    and expose `num`, `configs`, and the `DIR_*` lookups for the chart cells."""
    src = (
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        f"raw = pd.read_csv({json.dumps(csv_rel)})\n"
        "num = raw.copy()\n"
        "for c in ['asserts_passed', 'total_asserts', 'local_time_s', 'cost_usd', 'llm_calls']:\n"
        "    if c in num.columns:\n"
        "        num[c] = pd.to_numeric(num[c], errors='coerce')\n"
        "# Pass rate per execution; total_asserts == 0 -> undefined (NaN).\n"
        "num['pass_rate'] = num['asserts_passed'] / num['total_asserts'].replace(0, pd.NA)\n"
        "configs = sorted(num['config'].dropna().astype(str).unique())\n"
        "DIR_MARK = {'max': '\\u2191 higher is better', 'min': '\\u2193 lower is better'}\n"
        "print(f'{len(num)} executions across {len(configs)} configs')\n"
    )
    return new_code_cell(src)


def _key_chart_cell(col: str, label: str, direction: str) -> Any:
    """One diagram: grouped bars, one group per config, four bars per group for
    the interface (cli/sdk) × skills (with/without) combos, y = mean of `col`
    across all matching executions; dashed line = mean across all configs."""
    src = (
        f"col, label, direction = {json.dumps(col)}, {json.dumps(label)}, {json.dumps(direction)}\n"
        "# Four sub-bars per config: interface (cli/sdk) x skills (with/without).\n"
        "SUB = [('cli', 'none', 'cli / no-skills'), ('cli', 'official', 'cli / skills'),\n"
        "       ('sdk', 'none', 'sdk / no-skills'), ('sdk', 'official', 'sdk / skills')]\n"
        "piv = num.groupby(['config', 'interface', 'skills'])[col].mean()\n"
        "x = range(len(configs))\n"
        "nsub = len(SUB)\n"
        "width = 0.8 / nsub\n"
        "fig, ax = plt.subplots(figsize=(max(9, 2.4 * len(configs)), 5))\n"
        "colors = plt.cm.Paired.colors\n"
        "for j, (iface, sk, lab) in enumerate(SUB):\n"
        "    vals = [piv.get((cfg, iface, sk), float('nan')) for cfg in configs]\n"
        "    offs = [i + (j - (nsub - 1) / 2) * width for i in x]\n"
        "    ax.bar(offs, vals, width=width, label=lab, color=colors[j])\n"
        "    for o, v in zip(offs, vals):\n"
        "        if pd.notna(v):\n"
        "            ax.text(o, v, f'{v:.3g}', ha='center', va='bottom', fontsize=6, rotation=90)\n"
        "overall = num[col].mean()\n"
        "if pd.notna(overall):\n"
        "    ax.axhline(overall, color='tab:red', linestyle='--', linewidth=1,\n"
        "               label=f'all-config mean = {overall:.4g}')\n"
        "ax.set_xticks(list(x))\n"
        "ax.set_xticklabels(configs, rotation=20, ha='right', fontsize=9)\n"
        "ax.set_ylabel(label)\n"
        "ax.set_title(f'{label}  ({DIR_MARK[direction]})  \\u2014  mean across all tasks')\n"
        "ax.legend(title='interface / skills', fontsize=8)\n"
        "ax.grid(True, axis='y', alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
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

    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# Results — config comparison\n\n"
            f"Generated from `results.csv` — one row per execution across ALL "
            f"configs (**{len(rows)} execution(s)**). Each config is one "
            "experiment.\n\n"
            "Four diagrams, one per metric that counts. Each bar is a config, "
            "averaged across all of its executions (every task, interface, "
            "version, skills, repeat); the dashed line is the mean across all "
            "configs.\n\n"
            "- **pass rate** = `asserts_passed / total_asserts` — ↑ higher is better\n"
            "- **local_time_s** — ↓ lower is better\n"
            "- **cost_usd** — ↓ lower is better\n"
            "- **llm_calls** — ↓ lower is better\n"
        ),
        _setup_cell(csv_rel),
    ]
    for m in KEY_METRICS:
        nb.cells.append(new_markdown_cell(f"## {m['label']}"))
        nb.cells.append(_key_chart_cell(m["col"], m["label"], m["direction"]))

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
