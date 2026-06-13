"""Data-contract validation task (FTI sub-category: feature/validate) — generator.

Usage:
    python -m evals.feature.validate.generate --seed 7 --out /tmp/validate-7
    python -m evals.feature.validate.generate --selftest

An events export (data/events.csv) carries seeded contract violations at known
row ids — nulls in `amount`, amounts outside the documented range, categories
outside the documented enum. data/contract.md documents the rules. The agent
must load ONLY the clean rows into a feature table `events<sfx>` v1
(per-instance suffix) AND report the
rejected row ids in submission/answers.json: {"rejected": [row_id, ...]}.

Ground truth by construction: the generator seeded the violations. Naive
variants (gates assert they differ): ingesting everything; applying only the
null rule (range + enum violations kept).
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evals.common import canonicalize, digest, instance_suffix

ORIGIN = pd.Timestamp("2026-02-01", tz="UTC")
N_ROWS = 700
NULL_FRACTION = 0.03    # nulls in amount
RANGE_FRACTION = 0.02   # amounts outside [0, 10000]
ENUM_FRACTION = 0.02    # categories outside the enum
TABLE_BASE = "events"  # per-instance: f"{TABLE_BASE}{instance_suffix(seed)}"
CATEGORIES = ["grocery", "travel", "salary", "rent", "other"]
BAD_CATEGORIES = ["misc", "uncategorized", "promo??", "REFUND"]
AMOUNT_MIN, AMOUNT_MAX = 0.0, 10000.0

SPEC = {
    "columns": ["row_id", "account_id", "event_time", "amount", "category"],
    "ts_cols": ["event_time"],
    "int_cols": [],
    "sort_cols": ["row_id"],
}
VARIANT_DIAGNOSIS = {
    "ingest_everything": "contract violations were ingested (no rows rejected)",
    "wrong_rule": "only null amounts were rejected — range and enum violations were kept",
}


class GateError(RuntimeError):
    pass


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    table = TABLE_BASE + instance_suffix(seed)
    n = N_ROWS
    df = pd.DataFrame({
        "row_id": [f"R{i:05d}" for i in range(n)],
        "account_id": [f"A{int(a):04d}" for a in rng.integers(0, 150, n)],
        "event_time": [(ORIGIN + pd.Timedelta(minutes=int(m))).floor("s").strftime(
            "%Y-%m-%dT%H:%M:%SZ") for m in np.sort(rng.integers(0, 45 * 24 * 60, n))],
        "amount": np.round(rng.uniform(1.0, 9500.0, n), 2),
        "category": rng.choice(CATEGORIES, n),
    })

    # Seed disjoint violation sets at known row indices.
    n_null = int(n * NULL_FRACTION)
    n_range = int(n * RANGE_FRACTION)
    n_enum = int(n * ENUM_FRACTION)
    bad_idx = rng.choice(n, n_null + n_range + n_enum, replace=False)
    null_idx, range_idx, enum_idx = np.split(bad_idx, [n_null, n_null + n_range])

    df.loc[null_idx, "amount"] = np.nan
    low = rng.uniform(-900.0, -1.0, n_range)
    high = rng.uniform(AMOUNT_MAX + 1.0, 25000.0, n_range)
    df.loc[range_idx, "amount"] = np.round(
        np.where(rng.uniform(size=n_range) < 0.5, low, high), 2)
    df.loc[enum_idx, "category"] = rng.choice(BAD_CATEGORIES, n_enum)

    rejected_ids = sorted(df.loc[np.sort(bad_idx), "row_id"].tolist())
    clean_mask = (df["amount"].notna()
                  & df["amount"].between(AMOUNT_MIN, AMOUNT_MAX)
                  & df["category"].isin(CATEGORIES))
    truth = canonicalize(df.loc[clean_mask], SPEC)

    # --- gates ---------------------------------------------------------------
    if sorted(df.loc[~clean_mask, "row_id"]) != rejected_ids:
        raise GateError(f"reference rejection set disagrees with seeded ids (seed={seed})")
    variants = {
        "ingest_everything": canonicalize(df, SPEC),
        "wrong_rule": canonicalize(df.loc[df["amount"].notna()], SPEC),
    }
    for name, v in variants.items():
        if digest(v) == digest(truth):
            raise GateError(f"variant {name!r} matches truth (seed={seed})")
    # reference: apply all 3 documented rules == truth (independent re-read path)
    ref_df = pd.read_csv(io.StringIO(df.to_csv(index=False)))
    ref_mask = (ref_df["amount"].notna()
                & (ref_df["amount"] >= AMOUNT_MIN) & (ref_df["amount"] <= AMOUNT_MAX)
                & ref_df["category"].isin(CATEGORIES))
    if digest(canonicalize(ref_df.loc[ref_mask], SPEC)) != digest(truth):
        raise GateError(f"reference clean load disagrees with truth (seed={seed})")

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    df.to_csv(out / "data" / "events.csv", index=False)
    (out / "data" / "contract.md").write_text(
        "# Data contract — events export\n\n"
        "Columns: `row_id` (string, unique record key), `account_id` (string), "
        "`event_time` (ISO-8601 UTC timestamp), `amount` (double), `category` (string).\n\n"
        "A row is VALID only if ALL of the following hold:\n\n"
        "1. **amount is present** — null/empty amounts are contract violations.\n"
        f"2. **amount is within [{AMOUNT_MIN:.0f}, {AMOUNT_MAX:.0f}]** (inclusive).\n"
        "3. **category is one of**: " + ", ".join(f"`{c}`" for c in CATEGORIES) + ".\n\n"
        "Rows violating any rule must NOT be loaded; their ids must be reported.\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains an events export (data/events.csv) and the "
        "data contract it must satisfy (data/contract.md). Some rows violate the "
        "contract.\n"
        f"Register a feature table named `{table}`, version 1, on the platform, with "
        "record key `row_id` and event-timestamp column `event_time`, and load ONLY "
        "the rows that satisfy every contract rule.\n"
        "Make the table's features available for low-latency lookup as well "
        "(online/real-time access), where the platform distinguishes the two.\n"
        "Additionally write submission/answers.json listing every rejected row id:\n"
        '    {"rejected": ["<row_id>", ...]}\n'
    )
    meta = {
        "family": "validate", "seed": seed,
        "table_name": table, "table_version": 1,
        "spec": SPEC, "row_count": len(truth), "digest": digest(truth),
        "record_ids": truth["row_id"].tolist(),
        "rejected_ids": rejected_ids,
        "variant_digests": {k: digest(v) for k, v in variants.items()},
        "variant_diagnosis": VARIANT_DIAGNOSIS,
        "spot_rows": truth.head(3).to_dict(orient="records"),
    }
    (out / "solution" / "truth.json").write_text(json.dumps(meta, indent=2))
    (out / "instance.json").write_text(json.dumps({"family": "validate", "seed": seed}, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-validate-selftest/{seed}"))
            print(f"[validate] seed={seed} clean_rows={meta['row_count']} "
                  f"rejected={len(meta['rejected_ids'])} gates=OK")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
