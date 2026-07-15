"""PIT training-data task (FTI sub-category: feature/training_data) — generator.

Usage:
    python -m evals.feature.pit.generate --seed 7 --out /tmp/pit-7
    python -m evals.feature.pit.generate --selftest          # several seeds, gates

One task, no difficulty tiers: build a point-in-time-correct training dataset
across four feature tables whose histories contain late-arriving rows and
duplicates, including one table whose post-label rows strongly encode the
label (the leak temptation). The prompt states the as-of join rule explicitly
— what's measured is whether the agent can execute the discipline through the
platform, not whether it guesses the requirement.

Ground truth by construction: the generator creates the event streams, so it
knows the correct training dataset before any agent runs. Truth is computed by
an explicit per-label scan and cross-checked at generation time against the
independent `reference.pit_join` (merge_asof) implementation; every naive
variant must DIFFER from truth (discriminative gate), else generation raises
and the seed is rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evals.common import instance_suffix
from evals.feature.pit.reference import TABLE_FEATURES, latest_join, leaky_join, pit_join

KIND = "dataset"  # deliverable kind: table | dataset | answers | platform
SUMMARY = (
    "Measures whether an agent can build a point-in-time-correct training dataset across "
    "four feature tables with late-arriving rows, duplicates, and a leak-tempting table "
    "(the task family for the `training_data` FTI sub-category)."
)

ORIGIN = pd.Timestamp("2026-01-01", tz="UTC")
HORIZON_DAYS = 90
LABEL_WINDOW = (55, 75)  # label_time lands in this day range
DATASET_BASE = "churntraining"  # per-instance: f"{DATASET_BASE}{instance_suffix(seed)}"

# The one task configuration (no tiers).
KNOBS: dict = dict(
    n_accounts=100,
    tables=("transactions", "profiles", "activity", "account_health"),
    late=True,  # a later export of transactions rows must be ingested too
    dupes=True,  # exact duplicate rows (realism noise; value-identical)
    leak=True,  # account_health has post-label rows that encode the label
)
LEAK_TABLE = "account_health"
N_LATE_FORCED = 15  # accounts whose latest pre-label tx row arrives late
LATE_FRACTION = 0.10  # plus this share of other pre-label tx rows
DUPE_FRACTION = 0.05


# --------------------------------------------------------------------------
# World generation
# --------------------------------------------------------------------------


def _ts(day: float) -> pd.Timestamp:
    return (ORIGIN + pd.Timedelta(days=float(day))).floor("s")


def _gen_world(
    rng: np.random.Generator, knobs: dict
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return (labels, tables). Guarantees per account: >=1 row before
    label_time in every table (no-NaN truth) and >=1 transactions/profiles row
    after label_time (so the naive latest-join provably differs)."""
    accounts = [f"A{i:04d}" for i in range(knobs["n_accounts"])]
    label_day = rng.uniform(*LABEL_WINDOW, len(accounts))

    tx_rows, pr_rows, ac_rows, hl_rows = [], [], [], []
    for acct, ld in zip(accounts, label_day):
        # transactions: a few pre-label, at least one post-label
        days = sorted(
            [rng.uniform(0, 1)]  # forced early row
            + list(rng.uniform(2, ld - 0.5, rng.integers(3, 7)))  # pre-label
            + list(rng.uniform(ld + 1, HORIZON_DAYS, rng.integers(1, 3)))  # post-label
        )
        balance = float(rng.normal(5000, 2000))
        for d in days:
            amount = round(float(rng.lognormal(3.5, 1.0)), 2)
            balance = round(balance + float(rng.normal(0, 400)) - amount / 10, 2)
            tx_rows.append((acct, _ts(d), amount, balance))

        # profiles: initial row + one update (post-label for a third of accounts,
        # pre-label for another third — both shapes occur)
        score = int(rng.integers(300, 851))
        tier = str(rng.choice(["bronze", "silver", "gold"]))
        pr_rows.append((acct, _ts(rng.uniform(0, 1)), score, tier))
        u = rng.uniform()
        if u < 1 / 3:
            pr_rows.append(
                (
                    acct,
                    _ts(rng.uniform(2, ld - 1)),
                    int(np.clip(score + rng.integers(-80, 81), 300, 850)),
                    str(rng.choice(["bronze", "silver", "gold"])),
                )
            )
        elif u < 2 / 3:
            pr_rows.append(
                (
                    acct,
                    _ts(rng.uniform(ld + 1, HORIZON_DAYS)),
                    int(np.clip(score + rng.integers(-80, 81), 300, 850)),
                    str(rng.choice(["bronze", "silver", "gold"])),
                )
            )

        # activity: roughly weekly sessions counts across the horizon
        for d in np.arange(rng.uniform(0, 1), HORIZON_DAYS, 7.0):
            ac_rows.append((acct, _ts(d + rng.uniform(0, 1)), int(rng.integers(0, 40))))

    labels = pd.DataFrame(
        {
            "account_id": accounts,
            "label_time": [_ts(d) for d in label_day],
        }
    )

    tables = {
        "transactions": pd.DataFrame(
            tx_rows, columns=["account_id", "event_time", "amount", "balance"]
        ),
        "profiles": pd.DataFrame(
            pr_rows, columns=["account_id", "event_time", "credit_score", "tier"]
        ),
        "activity": pd.DataFrame(ac_rows, columns=["account_id", "event_time", "sessions_7d"]),
    }
    tables = {k: v for k, v in tables.items() if k in knobs["tables"]}

    # churn label: low balance at label time → likelier churn (realism only;
    # grading never depends on label semantics, just on passthrough).
    pit_balance = pit_join(labels, {"transactions": tables["transactions"]})["balance"]
    p = 1.0 / (1.0 + np.exp((pit_balance - pit_balance.median()) / 2000.0))
    labels["churned"] = (rng.uniform(size=len(labels)) < p).astype(int)

    # leak table: neutral value before labels; a strongly label-encoding value
    # AFTER label_time — the as-of-correct join sees only the neutral row.
    if knobs["leak"]:
        for acct, ld, ch in zip(accounts, label_day, labels["churned"]):
            hl_rows.append((acct, _ts(rng.uniform(0, 2)), round(float(rng.normal(50, 5)), 2)))
            post = float(rng.uniform(5, 15)) if ch else float(rng.uniform(85, 95))
            hl_rows.append((acct, _ts(ld + rng.uniform(3, 12)), round(post, 2)))
        tables[LEAK_TABLE] = pd.DataFrame(
            hl_rows, columns=["account_id", "event_time", "health_score"]
        )

    return labels, tables


def _split_late(
    rng: np.random.Generator, labels: pd.DataFrame, tx: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Withhold some PRE-LABEL transaction rows into a 'late export' file. For
    N_LATE_FORCED accounts the withheld row is their LATEST pre-label row, so
    ignoring the late file provably changes the result (discriminative)."""
    lt = labels.set_index("account_id")["label_time"]
    pre = tx[tx["event_time"] <= tx["account_id"].map(lt)]

    forced_accts = rng.choice(labels["account_id"], N_LATE_FORCED, replace=False)
    forced_idx = (
        pre[pre["account_id"].isin(forced_accts)]
        .sort_values("event_time")
        .groupby("account_id")
        .tail(1)
        .index
    )
    rest = pre.index.difference(forced_idx)
    extra_idx = rng.choice(rest, int(len(rest) * LATE_FRACTION), replace=False)
    late_idx = forced_idx.union(pd.Index(extra_idx))

    return tx.drop(index=late_idx), tx.loc[late_idx].sort_values(["account_id", "event_time"])


def _add_dupes(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    """Append exact duplicate rows (realism noise; value-identical, so any
    reasonable pipeline survives them — the load-bearing trap is the late file)."""
    dup = df.sample(frac=DUPE_FRACTION, random_state=int(rng.integers(0, 2**31)))
    return (
        pd.concat([df, dup], ignore_index=True)
        .sample(frac=1.0, random_state=int(rng.integers(0, 2**31)))
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# Truth + canonical form
# --------------------------------------------------------------------------


def _truth_scan(labels: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-label-row scan PIT join — the answer key, implemented independently
    of reference.pit_join (merge_asof) so the gate cross-checks both."""
    out_rows = []
    for _, lab in labels.iterrows():
        row = {
            "account_id": lab["account_id"],
            "label_time": lab["label_time"],
            "churned": lab["churned"],
        }
        for name, df in tables.items():
            sub = df[
                (df["account_id"] == lab["account_id"]) & (df["event_time"] <= lab["label_time"])
            ]
            best = sub.loc[sub["event_time"].idxmax()]
            for col in TABLE_FEATURES[name]:
                row[col] = best[col]
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def canonical_columns(tables: tuple[str, ...] | list[str]) -> list[str]:
    cols = ["account_id", "label_time"]
    for t in tables:
        cols += TABLE_FEATURES[t]
    return cols + ["churned"]


def canonicalize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normalize to the canonical comparable form: spec'd columns only, sorted
    by account_id, floats rounded 6 dp, timestamps as UTC ISO strings."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    out = out[columns].sort_values("account_id").reset_index(drop=True)
    for c in columns:
        if c == "label_time":
            out[c] = pd.to_numeric(out[c]).astype("int64")  # epoch-milliseconds (bigint)
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].astype(float).round(6)
        elif c in ("credit_score", "sessions_7d", "churned"):
            out[c] = out[c].astype("int64")
        elif pd.api.types.is_object_dtype(out[c]):
            out[c] = out[c].astype(str)
    return out


def digest(df: pd.DataFrame) -> str:
    return "sha256:" + hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------


def _prompt(knobs: dict, columns: list[str], dataset_name: str) -> str:
    files = "\n".join(f"  - data/{t}.csv" for t in knobs["tables"])
    if knobs["late"]:
        files += (
            "\n  - data/transactions_late.csv (a later export of transactions rows that "
            "arrived after the initial export — same schema, part of the same data)"
        )
    return (
        f"The directory data/ contains feature data for customer accounts:\n{files}\n"
        "  - data/labels.csv (account_id, label_time, churned)\n"
        "  - data/schema.md (column documentation)\n\n"
        f"Produce a versioned training dataset named `{dataset_name}`, version 1, on the platform, "
        f"with exactly these columns: {', '.join(columns)}. One row per row of labels.csv.\n"
        "For each (account_id, label_time) in labels.csv, every feature value must be the MOST "
        "RECENT value from its source table at or before label_time — values with event_time "
        "after label_time must not be used, in any table. `event_time` and `label_time` are "
        "epoch milliseconds. Join on account_id.\n"
    )


# --------------------------------------------------------------------------
# Instance assembly + gates
# --------------------------------------------------------------------------


class GateError(RuntimeError):
    pass


def generate(seed: int, out: Path) -> dict:
    knobs = KNOBS
    dataset_name = DATASET_BASE + instance_suffix(seed)
    rng = np.random.default_rng(seed)
    labels, tables = _gen_world(rng, knobs)
    columns = canonical_columns(knobs["tables"])

    # Encode all event/label times as epoch-MILLISECONDS (bigint) so the CLI can
    # ingest them directly as event-time columns (no client-side timestamp
    # parsing). Done after world generation (which uses Timestamps internally);
    # epoch-ms preserves ordering, so the as-of join/scan below is unaffected.
    def _to_ms(s: pd.Series) -> pd.Series:
        return pd.to_datetime(s, utc=True).dt.as_unit("ns").astype("int64") // 10**6

    labels["label_time"] = _to_ms(labels["label_time"])
    for _name in tables:
        tables[_name]["event_time"] = _to_ms(tables[_name]["event_time"])

    truth = canonicalize(_truth_scan(labels, tables), columns)

    # --- gates -------------------------------------------------------------
    ref = canonicalize(pit_join(labels, tables), columns)
    if digest(ref) != digest(truth):  # solvable + answer-key cross-check
        raise GateError("reference (merge_asof) disagrees with truth scan")

    variants: dict[str, pd.DataFrame] = {
        "latest_join": canonicalize(latest_join(labels, tables), columns),
    }
    if knobs["late"]:
        tx_main, tx_late = _split_late(rng, labels, tables["transactions"])
        without_late = {**tables, "transactions": tx_main}
        variants["ignore_late"] = canonicalize(pit_join(labels, without_late), columns)
    if knobs["leak"]:
        variants["leak_future"] = canonicalize(leaky_join(labels, tables, LEAK_TABLE), columns)

    for name, v in variants.items():
        if digest(v) == digest(truth):  # discriminative
            raise GateError(
                f"naive variant {name!r} matches truth — instance not discriminative (seed={seed})"
            )

    # --- write instance ------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()

    emit = dict(tables)
    if knobs["late"]:
        emit["transactions"] = tx_main
        tx_late.to_csv(out / "data" / "transactions_late.csv", index=False)
    if knobs["dupes"]:
        emit["transactions"] = _add_dupes(rng, emit["transactions"])
    for name, df in emit.items():
        df.sort_values(["account_id", "event_time"]).to_csv(
            out / "data" / f"{name}.csv", index=False
        )
    labels.to_csv(out / "data" / "labels.csv", index=False)

    schema = [
        "# Schema",
        "",
        "All tables join on `account_id`. `event_time` (bigint, epoch "
        "MILLISECONDS) is when the row became valid.",
        "",
    ]
    for name in emit:
        schema.append(
            f"- **{name}.csv**: account_id, event_time (epoch ms), "
            + ", ".join(TABLE_FEATURES[name])
        )
    schema.append("- **labels.csv**: account_id, label_time (epoch ms), churned (1 = churned)")
    (out / "data" / "schema.md").write_text("\n".join(schema) + "\n")

    (out / "prompt.txt").write_text(_prompt(knobs, columns, dataset_name))

    truth.to_csv(out / "solution" / "truth.csv", index=False)
    for name, v in variants.items():
        v.to_csv(out / "solution" / f"variant_{name}.csv", index=False)
    truth_meta = {
        "family": "pit",
        "seed": seed,
        "dataset_name": dataset_name,
        "dataset_version": 1,
        "columns": columns,
        "row_count": len(truth),
        "digest": digest(truth),
        "variant_digests": {k: digest(v) for k, v in variants.items()},
        "spot_rows": truth.head(3).to_dict(orient="records"),
    }
    (out / "solution" / "truth.json").write_text(json.dumps(truth_meta, indent=2))
    (out / "instance.json").write_text(
        json.dumps(
            {"family": "pit", "seed": seed, "knobs": {**knobs, "tables": list(knobs["tables"])}},
            indent=2,
        )
    )
    return truth_meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--selftest", action="store_true", help="generate several seeds into /tmp and report gates"
    )
    args = ap.parse_args(argv)

    if args.selftest:
        for seed in (1, 2, 3):
            out = Path(f"/tmp/mlpab-pit-selftest/{seed}")
            meta = generate(seed, out)
            print(
                f"[pit] seed={seed} rows={meta['row_count']:4d} "
                f"gates=OK variants={list(meta['variant_digests'])}"
            )
        return 0

    if not args.out:
        ap.error("--out is required (or use --selftest)")
    meta = generate(args.seed, args.out)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
