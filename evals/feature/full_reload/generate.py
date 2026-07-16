"""Full-reload task (FTI sub-category: feature/full_reload) — generator.

Usage:
    python -m evals.feature.full_reload.generate --seed 7 --out /tmp/full_reload-7
    python -m evals.feature.full_reload.generate --selftest

The upstream source made a BREAKING schema change: the initial export
(data/initial_export.csv — schema A: row_id, name, balance_eur, updated_at)
is superseded by a full re-export (data/reload/new_export.csv — schema B:
row_id, full_name, balance, currency, updated_at). Every surviving row is
re-issued with new values; some old row_ids are retired, new ones appear.
The table `customers<sfx>` (per-instance suffix) must be fully re-created:
version 1 from the initial
export, then version 2 containing EXACTLY the new export (no stale rows, no
old columns). The graded deliverable is version 2.

Ground truth by construction: truth = the new export. Naive variants (gates
assert they differ): merging instead of reloading (retired rows survive);
keeping the old column name for the balance (caught by A1_columns; its digest
is precomputed for diagnosis).
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

KIND = "table"  # deliverable kind: table | dataset | answers | platform
SUMMARY = (
    "Measures whether an agent can fully re-create a feature table after a breaking upstream "
    "schema change instead of merging the new export into the old data."
)

ORIGIN_OLD = pd.Timestamp("2026-01-15", tz="UTC")
ORIGIN_NEW = pd.Timestamp("2026-04-01", tz="UTC")
N_OLD = 180
N_RETIRED = (25, 40)  # seeded range of retired old row_ids
N_NEW = (30, 50)  # seeded range of newly added row_ids
TABLE_BASE = "customers"  # per-instance: f"{TABLE_BASE}{instance_suffix(seed)}"
CURRENCIES = ["EUR", "USD", "GBP", "SEK"]
FIRST = [
    "Ada",
    "Grace",
    "Alan",
    "Edsger",
    "Barbara",
    "Donald",
    "Margaret",
    "John",
    "Radia",
    "Vint",
    "Tim",
    "Frances",
]
LAST = [
    "Lovelace",
    "Hopper",
    "Turing",
    "Dijkstra",
    "Liskov",
    "Knuth",
    "Hamilton",
    "McCarthy",
    "Perlman",
    "Cerf",
    "Berners-Lee",
    "Allen",
]

SPEC = {  # schema B — the graded version-2 table
    "columns": ["row_id", "full_name", "balance", "currency", "updated_at"],
    "ts_cols": [],
    "int_cols": ["updated_at"],  # epoch-milliseconds (bigint)
    "sort_cols": ["row_id"],
}
SPEC_OLD_NAME = {  # schema B but the balance column kept its old name
    "columns": ["row_id", "full_name", "balance_eur", "currency", "updated_at"],
    "ts_cols": [],
    "int_cols": ["updated_at"],  # epoch-milliseconds (bigint)
    "sort_cols": ["row_id"],
}
VARIANT_DIAGNOSIS = {
    "merged": "the table was merged instead of re-created — retired rows survive",
    "old_schema_kept": "the balance column kept its old name (balance_eur)",
}


class GateError(RuntimeError):
    pass


def _names(rng: np.random.Generator, n: int) -> list[str]:
    return [
        f"{FIRST[int(i)]} {LAST[int(j)]}"
        for i, j in zip(rng.integers(0, len(FIRST), n), rng.integers(0, len(LAST), n))
    ]


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    table = TABLE_BASE + instance_suffix(seed)

    # Schema A — the initial export.
    old = pd.DataFrame(
        {
            "row_id": [f"C{i:05d}" for i in range(N_OLD)],
            "name": _names(rng, N_OLD),
            "balance_eur": np.round(rng.uniform(-2000.0, 50000.0, N_OLD), 2),
            # epoch-milliseconds (bigint) — ingested directly as event-time.
            "updated_at": [
                (ORIGIN_OLD + pd.Timedelta(minutes=int(m))).floor("s").value // 10**6
                for m in np.sort(rng.integers(0, 30 * 24 * 60, N_OLD))
            ],
        }
    )

    # Schema B — full re-export: every surviving row re-issued with new values,
    # some old ids retired, some new ids added.
    n_retired = int(rng.integers(*N_RETIRED))
    n_added = int(rng.integers(*N_NEW))
    retired = set(rng.choice(old["row_id"], n_retired, replace=False).tolist())
    surviving = [r for r in old["row_id"] if r not in retired]
    ids = surviving + [f"C{i:05d}" for i in range(N_OLD, N_OLD + n_added)]
    n = len(ids)
    new = (
        pd.DataFrame(
            {
                "row_id": ids,
                "full_name": _names(rng, n),
                "balance": np.round(rng.uniform(-2000.0, 50000.0, n), 2),
                "currency": rng.choice(CURRENCIES, n),
                "updated_at": [
                    (ORIGIN_NEW + pd.Timedelta(minutes=int(m))).floor("s").value // 10**6
                    for m in np.sort(rng.integers(0, 7 * 24 * 60, n))
                ],
            }
        )
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    truth = canonicalize(new, SPEC)

    # --- gates ---------------------------------------------------------------
    # merged: retired old rows survive, naively mapped into schema B (the
    # natural-but-wrong migration: name -> full_name, balance_eur -> balance,
    # currency assumed EUR), new rows win on overlapping ids.
    stale = (
        old[old["row_id"].isin(retired)]
        .rename(columns={"name": "full_name", "balance_eur": "balance"})
        .assign(currency="EUR")
    )
    variants = {
        "merged": canonicalize(pd.concat([new, stale], ignore_index=True), SPEC),
        "old_schema_kept": canonicalize(
            new.rename(columns={"balance": "balance_eur"}), SPEC_OLD_NAME
        ),
    }
    for name, v in variants.items():
        if digest(v) == digest(truth):
            raise GateError(f"variant {name!r} matches truth (seed={seed})")
    # reference: load exactly the new export == truth
    ref = canonicalize(new.sort_values("updated_at"), SPEC)
    if digest(ref) != digest(truth):
        raise GateError(f"reference reload disagrees with truth (seed={seed})")
    if not retired or not (set(old["row_id"]) - retired):
        raise GateError(f"degenerate retirement split (seed={seed})")

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data" / "reload").mkdir(parents=True)
    (out / "solution").mkdir()
    old.to_csv(out / "data" / "initial_export.csv", index=False)
    new.to_csv(out / "data" / "reload" / "new_export.csv", index=False)
    (out / "data" / "schema.md").write_text(
        "# Schema\n\n## initial_export.csv (the original schema)\n"
        "- **row_id** (string): unique record key\n- **name** (string)\n"
        "- **balance_eur** (double): balance, always in EUR\n"
        "- **updated_at** (bigint): epoch MILLISECONDS, the event-time column\n\n"
        "## reload/new_export.csv (the NEW, breaking schema)\n"
        "A complete re-export from the upstream source. Columns were renamed and "
        "extended; every row was re-issued with new values; some old row_ids no "
        "longer exist and new ones were added.\n"
        "- **row_id** (string): unique record key\n"
        "- **full_name** (string): replaces `name`\n"
        "- **balance** (double): replaces `balance_eur`; currency now varies\n"
        "- **currency** (string): ISO code\n"
        "- **updated_at** (bigint): epoch MILLISECONDS, the event-time column\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains an initial export of a customers table "
        "(data/initial_export.csv), a later FULL re-export after a breaking schema "
        "change upstream (data/reload/new_export.csv), and data/schema.md "
        "documenting both schemas.\n"
        f"First register a feature table named `{table}`, version 1, on the platform "
        "(record key `row_id`, event-time column `updated_at` in epoch milliseconds) "
        "and load the initial export into it.\n"
        "Then re-create the table from scratch for the new schema: a feature table "
        f"`{table}`, version 2 (record key `row_id`, event-time column "
        "`updated_at` in epoch milliseconds), containing EXACTLY the rows and columns "
        "of the new export — "
        "no stale rows from version 1, no old column names. Version 2 is the graded "
        "deliverable.\n"
        "Make version 2's features available for low-latency lookup as well "
        "(online/real-time access), where the platform distinguishes the two.\n"
    )
    meta = {
        "family": "full_reload",
        "seed": seed,
        "table_name": table,
        "table_version": 2,
        "spec": SPEC,
        "row_count": len(truth),
        "digest": digest(truth),
        "record_ids": truth["row_id"].tolist(),
        "retired_ids": sorted(retired),
        "variant_digests": {k: digest(v) for k, v in variants.items()},
        "variant_diagnosis": VARIANT_DIAGNOSIS,
        "spot_rows": truth.head(3).to_dict(orient="records"),
    }
    (out / "solution" / "truth.json").write_text(json.dumps(meta, indent=2))
    (out / "instance.json").write_text(
        json.dumps({"family": "full_reload", "seed": seed}, indent=2)
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
            meta = generate(seed, Path(f"/tmp/mlpab-full_reload-selftest/{seed}"))
            print(
                f"[full_reload] seed={seed} rows={meta['row_count']} "
                f"retired={len(meta['retired_ids'])} gates=OK"
            )
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
