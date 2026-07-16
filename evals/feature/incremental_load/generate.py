"""Incremental-load task (FTI sub-category: feature/incremental_load) — generator.

Usage:
    python -m evals.feature.incremental_load.generate --seed 7 --out /tmp/incload-7
    python -m evals.feature.incremental_load.generate --selftest

Six daily increment files (data/increment_01.csv .. increment_06.csv, no
overlapping rows) must ALL land in feature table `incremental<sfx>` v1, and a
RECURRING job/pipeline named `incrementaljob<sfx>` (both with the per-instance
suffix) must be registered on the platform to ingest future increments on a
schedule.

Ground truth by construction: truth = all increments concatenated. Naive
variant (gates assert it differs): skipping the last increment.
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

KIND = "platform"  # deliverable kind: table | dataset | answers | platform
SUMMARY = (
    "Measures whether an agent can load all daily increments into a feature table and register "
    "a recurring, scheduled ingestion job on the platform."
)

ORIGIN = pd.Timestamp("2026-03-02", tz="UTC")
N_INCREMENTS = 6
ROWS_PER_INCREMENT = (90, 130)  # seeded per-file row count range
TABLE_BASE = "incremental"  # per-instance: f"{TABLE_BASE}{instance_suffix(seed)}"
JOB_BASE = "incrementaljob"  # per-instance: f"{JOB_BASE}{instance_suffix(seed)}"
CATEGORIES = ["grocery", "travel", "salary", "rent", "other"]

SPEC = {
    "columns": ["row_id", "account_id", "event_time", "amount", "category"],
    "ts_cols": [],
    "int_cols": ["event_time"],  # epoch-milliseconds (bigint)
    "sort_cols": ["row_id"],
}
VARIANT_DIAGNOSIS = {
    "skip_last_increment": "the last increment file was never ingested",
}


class GateError(RuntimeError):
    pass


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    sfx = instance_suffix(seed)
    table = TABLE_BASE + sfx
    job_name = JOB_BASE + sfx
    counts = rng.integers(ROWS_PER_INCREMENT[0], ROWS_PER_INCREMENT[1] + 1, N_INCREMENTS)
    parts: list[pd.DataFrame] = []
    next_id = 0
    for day, cnt in enumerate(counts):
        n = int(cnt)
        day_origin = ORIGIN + pd.Timedelta(days=day)
        part = pd.DataFrame(
            {
                "row_id": [f"R{i:05d}" for i in range(next_id, next_id + n)],
                "account_id": [f"A{int(a):04d}" for a in rng.integers(0, 150, n)],
                # epoch-milliseconds (bigint) — the unit Hopsworks ingests
                # directly as an event-time column; no client-side parsing needed.
                "event_time": [
                    (day_origin + pd.Timedelta(minutes=int(m))).floor("s").value // 10**6
                    for m in np.sort(rng.integers(0, 24 * 60, n))
                ],
                "amount": np.round(rng.lognormal(3.0, 1.0, n), 2),
                "category": rng.choice(CATEGORIES, n),
            }
        )
        next_id += n
        parts.append(part)

    truth = canonicalize(pd.concat(parts, ignore_index=True), SPEC)

    # --- gates ---------------------------------------------------------------
    variants = {
        "skip_last_increment": canonicalize(pd.concat(parts[:-1], ignore_index=True), SPEC),
    }
    for name, v in variants.items():
        if digest(v) == digest(truth):
            raise GateError(f"variant {name!r} matches truth (seed={seed})")
    # reference: concat every increment, shuffled load order does not matter
    ref = canonicalize(
        pd.concat(
            [p.sample(frac=1.0, random_state=seed) for p in reversed(parts)], ignore_index=True
        ),
        SPEC,
    )
    if digest(ref) != digest(truth):
        raise GateError(f"reference concat disagrees with truth (seed={seed})")

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    for i, part in enumerate(parts, start=1):
        part.to_csv(out / "data" / f"increment_{i:02d}.csv", index=False)
    (out / "data" / "schema.md").write_text(
        "# Schema\n\nDaily increment files of one events table; each file holds one "
        "day's new rows (no overlaps between files). `row_id` uniquely identifies a "
        "row.\n\n- **row_id** (string): unique record key\n- **account_id** (string)\n"
        "- **event_time** (bigint): when the row became valid, as epoch "
        "MILLISECONDS — register it as the event-time column\n"
        "- **amount** (double)\n- **category** (string)\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains six daily increment files of an events table "
        f"(data/increment_01.csv .. data/increment_{N_INCREMENTS:02d}.csv) and "
        "data/schema.md documenting the columns. New increments with the same schema "
        "will keep arriving daily.\n"
        f"Register a feature table named `{table}`, version 1, on the platform, with "
        "record key `row_id` and event-time column `event_time` (epoch "
        "milliseconds), and load ALL provided increments into it.\n"
        "Make the table's features available for low-latency lookup as well "
        "(online/real-time access), where the platform distinguishes the two.\n"
        "Also set up a RECURRING job/pipeline on the platform, named "
        f"`{job_name}`, that would ingest future increments on a daily schedule "
        "(the schedule must be attached to the job, not just described).\n"
        "Finally write submission/answers.json as:\n"
        f'    {{"job_name": "{job_name}"}}\n'
    )
    meta = {
        "family": "incremental_load",
        "seed": seed,
        "table_name": table,
        "table_version": 1,
        "job_name": job_name,
        "spec": SPEC,
        "row_count": len(truth),
        "digest": digest(truth),
        "record_ids": truth["row_id"].tolist(),
        "variant_digests": {k: digest(v) for k, v in variants.items()},
        "variant_diagnosis": VARIANT_DIAGNOSIS,
        "spot_rows": truth.head(3).to_dict(orient="records"),
    }
    (out / "solution" / "truth.json").write_text(json.dumps(meta, indent=2))
    (out / "instance.json").write_text(
        json.dumps({"family": "incremental_load", "seed": seed}, indent=2)
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-incremental_load-selftest/{seed}"))
            print(f"[incremental_load] seed={seed} rows={meta['row_count']} gates=OK")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
