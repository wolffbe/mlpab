"""Ingest task (FTI sub-category: feature/ingest) — generator.

Usage:
    python -m evals.feature.ingest.generate --seed 7 --out /tmp/ingest-7
    python -m evals.feature.ingest.generate --selftest

Full load: register a feature table `transactions<sfx>` (`<sfx>` = the
6-hex per-instance suffix from the seed; record key `row_id`, event-time
column `event_time`) and load the provided export. One realistic
complication remains, stated in the schema doc: the export comes as TWO
files whose row ranges OVERLAP (the second is a re-delivery that includes
the tail of the first) — rows are identified by `row_id` and must land
exactly once. This is navigable on the baseline because the platform's
keyed upsert collapses the overlap on read; loading both files suffices.

`event_time` is delivered as epoch MILLISECONDS (a bigint) — the unit
Hopsworks accepts directly as an event-time column, so no client-side
timestamp parsing is required. (The earlier mixed ISO/epoch format was a
CLI-unparseable gotcha and was removed.) Ground truth is the rows made here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evals.common import canonicalize, digest, instance_suffix

ORIGIN = pd.Timestamp("2026-01-01", tz="UTC")
N_ROWS = 600
OVERLAP = 80  # rows re-delivered in the second file
TABLE_BASE = "transactions"  # per-instance: f"{TABLE_BASE}{instance_suffix(seed)}"

SPEC = {
    "columns": ["row_id", "account_id", "event_time", "amount", "category"],
    "ts_cols": [],
    "int_cols": ["event_time"],  # epoch-milliseconds (bigint)
    "sort_cols": ["row_id"],
}
VARIANT_DIAGNOSIS = {
    "keep_overlap": "the re-delivered overlap rows were loaded twice (dedupe by row_id)",
}


class GateError(RuntimeError):
    pass


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    table = TABLE_BASE + instance_suffix(seed)
    n = N_ROWS
    times = [
        (ORIGIN + pd.Timedelta(minutes=int(m))).floor("s")
        for m in np.sort(rng.integers(0, 60 * 24 * 60, n))
    ]
    # event_time as epoch-milliseconds (bigint) — the unit Hopsworks ingests
    # directly as an event-time column; no client-side parsing needed.
    event_ms = pd.DatetimeIndex(times).as_unit("ns").astype("int64") // 10**6
    truth_df = pd.DataFrame(
        {
            "row_id": [f"R{i:05d}" for i in range(n)],
            "account_id": [f"A{int(a):04d}" for a in rng.integers(0, 120, n)],
            "event_time": event_ms.astype("int64"),
            "amount": np.round(rng.lognormal(3.0, 1.0, n), 2),
            "category": rng.choice(["grocery", "travel", "salary", "rent", "other"], n),
        }
    )

    # Two exports whose row ranges OVERLAP: the second is a re-delivery that
    # includes the tail of the first. `row_id` is the record key, so the
    # platform's keyed upsert collapses the overlap on read — both CLI and SDK
    # navigate this by loading both files (no client-side dedupe required).
    split = n - OVERLAP - int(rng.integers(150, 250))
    part1 = truth_df.iloc[: split + OVERLAP]  # ...includes the overlap tail
    part2 = truth_df.iloc[split:].sample(frac=1.0, random_state=seed)  # re-delivery, shuffled

    truth = canonicalize(truth_df, SPEC)

    # --- gates ---------------------------------------------------------------
    # Diagnostic variant: keeping the re-delivered overlap duplicated.
    variants = {
        "keep_overlap": canonicalize(
            pd.concat([part1, part2], ignore_index=True).sort_values(["row_id", "event_time"]),
            {**SPEC, "sort_cols": ["row_id"]},
        ),
    }
    for name, v in variants.items():
        if digest(v) == digest(truth):
            raise GateError(f"variant {name!r} matches truth (seed={seed})")
    # reference: load both files, dedupe by row_id == truth (what the platform does)
    ref = canonicalize(
        pd.concat([part1, part2], ignore_index=True).drop_duplicates(subset="row_id"),
        SPEC,
    )
    if digest(ref) != digest(truth):
        raise GateError(f"reference load disagrees with truth (seed={seed})")

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    part1.to_csv(out / "data" / "transactions_export_1.csv", index=False)
    part2.to_csv(out / "data" / "transactions_export_2.csv", index=False)
    (out / "data" / "schema.md").write_text(
        "# Schema\n\nBoth export files share one schema; together they contain the "
        "full table, but their row ranges OVERLAP (the second file is a "
        "re-delivery that includes the tail of the first). `row_id` uniquely "
        "identifies a row.\n\n"
        "- **row_id** (string): unique record key\n"
        "- **account_id** (string)\n"
        "- **event_time** (bigint): when the row became valid, as epoch "
        "MILLISECONDS — register it as the event-time column\n"
        "- **amount** (double)\n- **category** (string)\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains two export files of a transactions table "
        "(data/transactions_export_1.csv, data/transactions_export_2.csv) and "
        "data/schema.md documenting the columns and their quirks.\n"
        f"Register a feature table named `{table}`, version 1, on the platform, with "
        "record key `row_id` and event-time column `event_time` (epoch "
        "milliseconds), and load the full export into it: every row exactly once.\n"
        "Make the table's features available for low-latency lookup as well "
        "(online/real-time access), where the platform distinguishes the two.\n"
    )
    meta = {
        "family": "ingest",
        "seed": seed,
        "table_name": table,
        "table_version": 1,
        "spec": SPEC,
        "row_count": len(truth),
        "digest": digest(truth),
        "record_ids": truth["row_id"].tolist(),
        "variant_digests": {k: digest(v) for k, v in variants.items()},
        "variant_diagnosis": VARIANT_DIAGNOSIS,
        "spot_rows": truth.head(3).to_dict(orient="records"),
    }
    (out / "solution" / "truth.json").write_text(json.dumps(meta, indent=2))
    (out / "instance.json").write_text(json.dumps({"family": "ingest", "seed": seed}, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-ingest-selftest/{seed}"))
            print(f"[ingest] seed={seed} rows={meta['row_count']} gates=OK")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
