"""Model-independent transformations task (feature/mit) — generator.

Usage:
    python -m evals.feature.mit.generate --seed 7 --out /tmp/mit-7
    python -m evals.feature.mit.generate --selftest

Given a seeded transactions export, produce a derived feature table
`features<sfx>` (per-instance suffix) with three documented transformations:

    amount_usd  = amount * fx_rate(currency)        (rates in data/fx_rates.csv)
    is_weekend  = 1 if event_time falls on Sat/Sun (UTC), else 0
    amount_7d   = sum of THIS account's `amount` over the 7 days up to and
                  including event_time (window [event_time-7d, event_time],
                  inclusive on both ends — a closed-closed window so a Trino
                  RANGE frame on the CLI baseline reproduces it exactly)

The window task is where naive implementations diverge: the gates assert that
an 8-day window and a window EXCLUDING the current row both differ from truth.
Truth is computed by a per-row scan and cross-checked against an independent
vectorized implementation. event_time is epoch-milliseconds (bigint) so the
CLI can ingest it directly as the event-time column.
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
    "Measures whether an agent can compute model-independent transformations, in particular a "
    "per-account rolling 7-day sum with exact window semantics, into a derived feature table."
)

ORIGIN = pd.Timestamp("2026-03-01", tz="UTC")
N_ROWS = 700
N_ACCOUNTS = 60
TABLE_BASE = "features"  # per-instance: f"{TABLE_BASE}{instance_suffix(seed)}"

SPEC = {
    "columns": ["row_id", "account_id", "event_time", "amount_usd", "is_weekend", "amount_7d"],
    "ts_cols": [],
    "int_cols": ["is_weekend", "event_time"],  # event_time = epoch-milliseconds (bigint)
    "sort_cols": ["row_id"],
}
VARIANT_DIAGNOSIS = {
    "window_8d": "the rolling window spans 8 days instead of 7",
    "window_excl_self": "the rolling sum excludes the current row (it must be included)",
}
FX = {"EUR": 1.09, "SEK": 0.095, "GBP": 1.27, "USD": 1.0}


def _scan_7d(df: pd.DataFrame, days: int = 7, include_self: bool = True) -> pd.Series:
    """Per-row scan: sum of the account's amounts in [t-days, t] (closed-closed).

    The lower bound is INCLUSIVE so a Trino `RANGE BETWEEN <days> PRECEDING AND
    CURRENT ROW` window frame reproduces it bit-for-bit on the CLI baseline (a
    half-open window is not expressible with a SQL RANGE frame).
    """
    out = np.zeros(len(df))
    for acct, grp in df.groupby("account_id"):
        for i, row in grp.iterrows():
            lo = row["event_time"] - pd.Timedelta(days=days)
            m = (grp["event_time"] >= lo) & (grp["event_time"] <= row["event_time"])
            if not include_self:
                m &= grp.index != i
            out[df.index.get_loc(i)] = grp.loc[m, "amount"].sum()
    return pd.Series(np.round(out, 2), index=df.index)


def _rolling_7d(df: pd.DataFrame, days: int = 7) -> pd.Series:
    """Independent vectorized implementation (pandas time-based rolling),
    closed-closed to match `_scan_7d`."""
    # rolling is positional within groups sorted by time; map back via order
    df_sorted = df.sort_values(["account_id", "event_time"])
    vals = (
        df_sorted.set_index("event_time")
        .groupby("account_id")["amount"]
        .rolling(f"{days}D", closed="both")
        .sum()
        .to_numpy()
    )
    out = pd.Series(np.round(vals, 2), index=df_sorted.index)
    return out.reindex(df.index)


class GateError(RuntimeError):
    pass


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    table = TABLE_BASE + instance_suffix(seed)
    n = N_ROWS
    df = pd.DataFrame(
        {
            "row_id": [f"R{i:05d}" for i in range(n)],
            "account_id": [f"A{int(a):04d}" for a in rng.integers(0, N_ACCOUNTS, n)],
            "event_time": sorted(
                (ORIGIN + pd.Timedelta(minutes=int(m))).floor("s")
                for m in rng.integers(0, 60 * 24 * 45, n)
            ),
            "amount": np.round(rng.lognormal(3.0, 1.0, n), 2),
            "currency": rng.choice(list(FX), n),
        }
    )

    derived = df[["row_id", "account_id", "event_time"]].copy()
    derived["amount_usd"] = np.round(df["amount"] * df["currency"].map(FX), 6)
    derived["is_weekend"] = (df["event_time"].dt.dayofweek >= 5).astype(int)
    derived["amount_7d"] = _scan_7d(df)
    # event_time emitted/graded as epoch-milliseconds (bigint); the derived
    # computations above used the Timestamp form.
    derived["event_time"] = derived["event_time"].dt.as_unit("ns").astype("int64") // 10**6
    truth = canonicalize(derived, SPEC)

    # --- gates ---------------------------------------------------------------
    ref = canonicalize(derived.assign(amount_7d=_rolling_7d(df)), SPEC)
    if digest(ref) != digest(truth):
        raise GateError(f"vectorized reference disagrees with scan (seed={seed})")
    variants = {
        "window_8d": canonicalize(derived.assign(amount_7d=_scan_7d(df, days=8)), SPEC),
        "window_excl_self": canonicalize(
            derived.assign(amount_7d=_scan_7d(df, include_self=False)), SPEC
        ),
    }
    for name, v in variants.items():
        if digest(v) == digest(truth):
            raise GateError(f"variant {name!r} matches truth (seed={seed})")

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    emit = df.copy()
    emit["event_time"] = emit["event_time"].dt.as_unit("ns").astype("int64") // 10**6
    emit.to_csv(out / "data" / "transactions.csv", index=False)
    pd.DataFrame({"currency": list(FX), "fx_rate": [FX[c] for c in FX]}).to_csv(
        out / "data" / "fx_rates.csv", index=False
    )
    (out / "data" / "schema.md").write_text(
        "# Schema\n\n- **transactions.csv**: row_id (unique key), account_id, "
        "event_time (bigint, epoch MILLISECONDS), amount, currency\n"
        "- **fx_rates.csv**: currency, fx_rate (multiply amount by this to get USD)\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains a transactions export (data/transactions.csv) "
        "and currency rates (data/fx_rates.csv).\n"
        f"Produce a derived feature table named `{table}`, version 1, on the platform, "
        "with record key `row_id`, event-time column `event_time` (epoch "
        "milliseconds), and exactly these columns: row_id, account_id, event_time, "
        "amount_usd, is_weekend, amount_7d, where:\n"
        "  - amount_usd = amount * fx_rate of the row's currency;\n"
        "  - is_weekend = 1 if event_time falls on Saturday or Sunday (UTC), else 0;\n"
        "  - amount_7d = the sum of THIS account's `amount` over the 7 days up to and "
        "including the row's event_time (the window is [event_time - 7 days, "
        "event_time], inclusive on both ends).\n"
        "One row per input row. Make the table's features available for low-latency "
        "lookup as well (online/real-time access), where the platform distinguishes "
        "the two.\n"
    )
    meta = {
        "family": "mit",
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
    (out / "instance.json").write_text(json.dumps({"family": "mit", "seed": seed}, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-mit-selftest/{seed}"))
            print(f"[mit] seed={seed} rows={meta['row_count']} gates=OK")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
