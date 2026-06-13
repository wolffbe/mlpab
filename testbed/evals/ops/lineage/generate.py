"""Lineage task (FTI sub-category: ops/lineage) — generator.

Usage:
    python -m evals.ops.lineage.generate --seed 7 --out /tmp/lineage-7
    python -m evals.ops.lineage.generate --selftest

World: two raw source tables (`data/raw_a.csv`: row_id, a_val and
`data/raw_b.csv`: row_id, b_val) whose row_id sets only PARTIALLY overlap,
plus the documented derivation rule: the derived table contains exactly the
row_ids present in BOTH sources, with col_sum = round(a_val + b_val, 6).
The agent must create feature tables `rawa<sfx>` v1 and `rawb<sfx>` v1, the
derived table `derived<sfx>` v1 (per-instance suffix; record key row_id;
columns row_id, col_sum), and answer the lineage question in
submission/answers.json:

    {"derived_from": ["rawa<sfx>", "rawb<sfx>"]}

Ground truth by construction (the generator seeds both sources and the
overlap). Naive variant (gates assert it differs): outer_join_fill0 — keeping
row_ids that exist in only one source and filling the missing value with 0.
Generation-time gates also run the grade function: the reference deliverables
pass; a spoiled col_sum and a wrong derived_from both fail.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from evals.common import canonicalize, digest, instance_suffix

N_IDS = 320            # universe of row ids
N_A = 250              # ids present in raw_a
N_B = 240              # ids present in raw_b (overlap is partial by construction)
# Per-instance platform table names: f"{base}{instance_suffix(seed)}" (the
# local data/raw_a.csv / data/raw_b.csv filenames stay fixed).
TABLE_BASE = "derived"
SOURCE_BASES = ["rawa", "rawb"]

SPEC = {
    "columns": ["row_id", "col_sum"],
    "ts_cols": [],
    "int_cols": [],
    "float_cols": ["col_sum"],
    "sort_cols": ["row_id"],
}
VARIANT_DIAGNOSIS = {
    "outer_join_fill0": "row_ids present in only ONE source were kept with the "
                        "missing value treated as 0 — the derived table must "
                        "contain only row_ids present in BOTH sources",
}


class GateError(RuntimeError):
    pass


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    sfx = instance_suffix(seed)
    table = TABLE_BASE + sfx
    sources = [base + sfx for base in SOURCE_BASES]
    ids = np.array([f"R{i:05d}" for i in range(N_IDS)])
    a_ids = np.sort(rng.choice(ids, N_A, replace=False))
    b_ids = np.sort(rng.choice(ids, N_B, replace=False))
    raw_a = pd.DataFrame({"row_id": a_ids,
                          "a_val": np.round(rng.normal(10, 4, N_A), 4)})
    raw_b = pd.DataFrame({"row_id": b_ids,
                          "b_val": np.round(rng.normal(-3, 5, N_B), 4)})
    both = sorted(set(a_ids) & set(b_ids))
    only_one = (set(a_ids) ^ set(b_ids))
    if len(both) < 50 or not only_one:
        raise GateError(f"degenerate overlap (seed={seed}): "
                        f"{len(both)} shared, {len(only_one)} exclusive")

    inner = raw_a.merge(raw_b, on="row_id", how="inner")
    truth_df = pd.DataFrame({"row_id": inner["row_id"],
                             "col_sum": np.round(inner["a_val"] + inner["b_val"], 6)})
    truth = canonicalize(truth_df, SPEC)

    # --- gates ---------------------------------------------------------------
    # independent reference: dict-based scan over the intersection
    a_map = dict(zip(raw_a["row_id"], raw_a["a_val"]))
    b_map = dict(zip(raw_b["row_id"], raw_b["b_val"]))
    ref = canonicalize(pd.DataFrame(
        [{"row_id": r, "col_sum": round(a_map[r] + b_map[r], 6)} for r in both]), SPEC)
    if digest(ref) != digest(truth):
        raise GateError(f"scan reference disagrees with inner join (seed={seed})")
    outer = raw_a.merge(raw_b, on="row_id", how="outer").fillna(0.0)
    variants = {"outer_join_fill0": canonicalize(pd.DataFrame({
        "row_id": outer["row_id"],
        "col_sum": np.round(outer["a_val"] + outer["b_val"], 6)}), SPEC)}
    for name, v in variants.items():
        if digest(v) == digest(truth):
            raise GateError(f"variant {name!r} matches truth (seed={seed})")

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    raw_a.to_csv(out / "data" / "raw_a.csv", index=False)
    raw_b.to_csv(out / "data" / "raw_b.csv", index=False)
    (out / "data" / "schema.md").write_text(
        "# Schema\n\n"
        "- **raw_a.csv**: row_id (unique key), a_val (double)\n"
        "- **raw_b.csv**: row_id (unique key), b_val (double)\n\n"
        "The two files cover different (overlapping) sets of row_ids.\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains two raw source tables (data/raw_a.csv: "
        "row_id, a_val and data/raw_b.csv: row_id, b_val); their row_id sets "
        "only partially overlap.\n"
        f"Create feature tables on the platform: `{sources[0]}`, version 1 (record key "
        f"row_id) loaded from raw_a.csv, and `{sources[1]}`, version 1 (record key "
        "row_id) loaded from raw_b.csv.\n"
        f"Then create the derived feature table `{table}`, version 1, with "
        "record key `row_id` and exactly these columns: row_id, col_sum — "
        f"containing ONLY the row_ids present in BOTH `{sources[0]}` and `{sources[1]}`, with "
        "col_sum = a_val + b_val rounded to 6 decimal places. Where the "
        "platform records lineage/provenance, register the derivation so "
        f"`{table}` is traceable to its sources.\n"
        "Make the derived table's features available for low-latency lookup as "
        "well (online/real-time access), where the platform distinguishes the "
        "two.\n"
        "Finally answer the lineage question in submission/answers.json:\n"
        '    {"derived_from": ["<source table name>", ...]}\n'
        f"— the sorted list of source tables `{table}` was derived from.\n"
        f"If no platform is available, also write the derived table to "
        f"submission/{table}.csv.\n"
    )
    meta = {
        "family": "lineage", "seed": seed,
        "table_name": table, "table_version": 1,
        "source_tables": sources,
        "spec": SPEC, "row_count": len(truth), "digest": digest(truth),
        "record_ids": truth["row_id"].tolist(),
        "variant_digests": {k: digest(v) for k, v in variants.items()},
        "variant_diagnosis": VARIANT_DIAGNOSIS,
        "spot_rows": truth.head(3).to_dict(orient="records"),
    }
    (out / "solution" / "truth.json").write_text(json.dumps(meta, indent=2))
    (out / "instance.json").write_text(json.dumps({"family": "lineage", "seed": seed}, indent=2))

    _gate_grader(out, truth_df, table, sources)
    return meta


def _gate_grader(inst: Path, truth_df: pd.DataFrame, table: str,
                 sources: list[str]) -> None:
    """Generation-time gates through the actual grade function (adapter none)."""
    from evals.ops.lineage.grade import grade

    def run(df: pd.DataFrame, answers: dict | None) -> bool:
        with tempfile.TemporaryDirectory(prefix="mlpab-lineage-gate-") as td:
            run_dir = Path(td)
            (run_dir / "submission").mkdir()
            df.to_csv(run_dir / "submission" / f"{table}.csv", index=False)
            if answers is not None:
                (run_dir / "submission" / "answers.json").write_text(json.dumps(answers))
            return grade(inst, "none", run_dir)["success"]

    if not run(truth_df, {"derived_from": sources}):
        raise GateError("reference deliverables fail the grade function")
    spoiled = truth_df.copy()
    spoiled.loc[spoiled.index[0], "col_sum"] += 1.0
    if run(spoiled, {"derived_from": sources}):
        raise GateError("spoiled derived table passes the grade function")
    if run(truth_df, {"derived_from": sources[:1]}):
        raise GateError("wrong derived_from answer passes the grade function")
    if run(truth_df, None):
        raise GateError("missing answers.json passes the grade function")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-lineage-selftest/{seed}"))
            print(f"[lineage] seed={seed} rows={meta['row_count']} gates=OK")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
