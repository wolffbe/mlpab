"""Backfill task (FTI sub-category: feature/backfill) — generator.

Usage:
    python -m evals.feature.backfill.generate --seed 7 --out /tmp/backfill-7
    python -m evals.feature.backfill.generate --selftest

Out-of-order batches including CORRECTIONS: three batch files of an accounts
table (platform name `accounts<sfx>`, the per-instance suffix) arrive in the
wrong order; the same `row_id` may appear in several
batches, and the row with the LATEST `updated_at` is the correct final state.
The agent must load them so the feature table's final contents equal that
latest-revision state.

Naive variants (gates assert they differ): concatenating everything (duplicate
row_ids survive); processing batches in FILE order with first-write-wins
(corrections lost).
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

ORIGIN = pd.Timestamp("2026-02-01", tz="UTC")
N_ROWS = 400
N_CORRECTED = 90  # rows that receive a later correction
TABLE_BASE = "accounts"  # per-instance: f"{TABLE_BASE}{instance_suffix(seed)}"

SPEC = {
    "columns": ["row_id", "status", "balance", "updated_at"],
    "ts_cols": ["updated_at"],
    "int_cols": [],
    "sort_cols": ["row_id"],
}
VARIANT_DIAGNOSIS = {
    "concat_all": "corrected rows were loaded twice — duplicates by row_id survive",
    "load_order_upsert": "batches were upserted in file order (last write wins by "
    "LOAD order, not by updated_at) — the out-of-order "
    "corrections were overwritten by older rows",
}


class GateError(RuntimeError):
    pass


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    table = TABLE_BASE + instance_suffix(seed)

    base = pd.DataFrame(
        {
            "row_id": [f"R{i:05d}" for i in range(N_ROWS)],
            "status": rng.choice(["active", "dormant", "closed"], N_ROWS),
            "balance": np.round(rng.normal(5000, 2000, N_ROWS), 2),
            "updated_at": [
                (ORIGIN + pd.Timedelta(minutes=int(m))).floor("s")
                for m in rng.integers(0, 10_000, N_ROWS)
            ],
        }
    )
    corrected_ids = rng.choice(N_ROWS, N_CORRECTED, replace=False)
    corrections = base.iloc[corrected_ids].copy()
    corrections["status"] = rng.choice(["active", "dormant", "closed"], N_CORRECTED)
    corrections["balance"] = np.round(corrections["balance"] + rng.normal(0, 800, N_CORRECTED), 2)
    corrections["updated_at"] = corrections["updated_at"] + pd.to_timedelta(
        rng.integers(60, 5_000, N_CORRECTED), unit="m"
    )

    # truth: latest updated_at per row_id
    full = pd.concat([base, corrections], ignore_index=True)
    truth_df = full.sort_values("updated_at").groupby("row_id", as_index=False).tail(1)
    truth = canonicalize(truth_df, SPEC)

    # batches, delivered OUT OF ORDER: batch_1 contains the CORRECTIONS (latest
    # data, delivered first), batches 2+3 the original load split in two.
    halves = np.array_split(base.sample(frac=1.0, random_state=seed), 2)
    batches = {
        "batch_1.csv": corrections.sample(frac=1.0, random_state=seed),
        "batch_2.csv": halves[0],
        "batch_3.csv": halves[1],
    }

    # --- gates ---------------------------------------------------------------
    concat_all = canonicalize(full.sort_values(["row_id", "updated_at"]), SPEC)
    in_load_order = pd.concat(list(batches.values()), ignore_index=True)
    load_order_upsert = canonicalize(
        in_load_order.drop_duplicates(subset="row_id", keep="last"), SPEC
    )
    variants = {"concat_all": concat_all, "load_order_upsert": load_order_upsert}
    for name, v in variants.items():
        if digest(v) == digest(truth):
            raise GateError(f"variant {name!r} matches truth (seed={seed})")
    ref = canonicalize(
        pd.concat(list(batches.values()), ignore_index=True)
        .sort_values("updated_at")
        .groupby("row_id", as_index=False)
        .tail(1),
        SPEC,
    )
    if digest(ref) != digest(truth):
        raise GateError(f"reference latest-revision load disagrees with truth (seed={seed})")

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    for name, df in batches.items():
        emit = df.copy()
        emit["updated_at"] = emit["updated_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        emit.to_csv(out / "data" / name, index=False)
    (out / "data" / "schema.md").write_text(
        "# Schema\n\nThree batch files of one `accounts` table, delivered OUT OF "
        "ORDER: the same `row_id` may appear in more than one batch, and the row "
        "with the LATEST `updated_at` is the correct, current state (later rows "
        "are corrections of earlier ones).\n\n"
        "- **row_id** (string): unique record key\n"
        "- **status** (string): active | dormant | closed\n"
        "- **balance** (double)\n"
        "- **updated_at**: revision timestamp (ISO-8601 UTC)\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains three batch files of an `accounts` table "
        "(data/batch_1.csv, data/batch_2.csv, data/batch_3.csv — see data/schema.md; "
        "they arrived out of order and contain corrections keyed by `row_id`).\n"
        f"Register a feature table named `{table}`, version 1, on the platform, with "
        "record key `row_id` and event-timestamp column `updated_at`, and load the "
        "batches so the table's final contents are each row's LATEST revision — "
        "exactly one row per row_id.\n"
        "Make the table's features available for low-latency lookup as well "
        "(online/real-time access), where the platform distinguishes the two.\n"
    )
    meta = {
        "family": "backfill",
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
    (out / "instance.json").write_text(json.dumps({"family": "backfill", "seed": seed}, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-backfill-selftest/{seed}"))
            print(f"[backfill] seed={seed} rows={meta['row_count']} gates=OK")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
