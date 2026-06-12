"""F5 (PIT) — reference and naive join implementations.

Deliberately a SEPARATE implementation path from the generator's truth scan
(`generate._truth_scan` walks label rows one by one; this module uses
`pd.merge_asof`). The generation-time gate asserts the two agree, so a bug in
either one fails generation instead of silently producing a wrong answer key.

`pit_join`   — the reference solution: most recent row at or before label_time.
`latest_join`— naive baseline: overall latest row per account (uses the future).
Both take `tables` as {name: dataframe-with(account_id, event_time, features…)}.
"""
from __future__ import annotations

import pandas as pd

# Feature columns each table contributes to the training dataset.
TABLE_FEATURES: dict[str, list[str]] = {
    "transactions": ["amount", "balance"],
    "profiles": ["credit_score", "tier"],
    "activity": ["sessions_7d"],
    "account_health": ["health_score"],
}


def _features(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    cols = ["account_id", "event_time"] + TABLE_FEATURES[table_name]
    return df[cols]


def pit_join(labels: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Point-in-time-correct join: for every (account_id, label_time), the most
    recent feature row with event_time <= label_time, per table."""
    out = labels.sort_values("label_time").reset_index(drop=True)
    for name, df in tables.items():
        feats = _features(name, df).sort_values("event_time")
        out = pd.merge_asof(
            out,
            feats,
            left_on="label_time",
            right_on="event_time",
            by="account_id",
            direction="backward",
            allow_exact_matches=True,
        ).drop(columns=["event_time"])
    return out


def latest_join(labels: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Naive join: overall latest row per account, ignoring label_time — the
    classic leak (uses data from the future)."""
    out = labels.copy()
    for name, df in tables.items():
        feats = _features(name, df)
        last = (
            feats.sort_values("event_time")
            .groupby("account_id", as_index=False)
            .tail(1)
            .drop(columns=["event_time"])
        )
        out = out.merge(last, on="account_id", how="left")
    return out


def leaky_join(
    labels: pd.DataFrame, tables: dict[str, pd.DataFrame], leak_table: str
) -> pd.DataFrame:
    """PIT-correct for every table EXCEPT `leak_table`, which is joined with the
    overall-latest rule — the tempting hard-tier mistake (the future row of the
    leak table visibly 'improves' the signal)."""
    clean = {k: v for k, v in tables.items() if k != leak_table}
    out = pit_join(labels, clean)
    return latest_join(out, {leak_table: tables[leak_table]})
