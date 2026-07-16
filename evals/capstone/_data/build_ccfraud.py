"""Build the ccfraud raw fixture — ONE-TIME, OFFLINE, KEYLESS.

Credit-card fraud in the mlfs-book is driven by SYNTHETIC transactions (the
book streams them through Feldera). We generate an equivalent deterministic
synthetic transaction log here and commit it as `ccfraud_raw.csv`, so the
testbed never needs a data source at run time — it just reads the CSV.

Fraud is LEARNABLE from engineered features (amount, geo distance from the
card's home, transaction velocity, odd hours) but NOT from any single raw
column — so the agent has to actually build a feature pipeline.

    python -m evals.capstone._data.build_ccfraud           # writes ccfraud_raw.csv
    python -m evals.capstone._data.build_ccfraud --rows 30000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "ccfraud_raw.csv"
ORIGIN = pd.Timestamp("2025-09-01", tz="UTC")
CATEGORIES = [
    "grocery",
    "restaurant",
    "fuel",
    "electronics",
    "travel",
    "entertainment",
    "health",
    "clothing",
    "online",
    "cash_advance",
]
SEED = 20240917  # fixed: the committed fixture must be stable across rebuilds


def build(n_cards: int, days: int, rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    # Each card has a home location and a baseline spend profile.
    home_lat = rng.uniform(25.0, 49.0, n_cards)
    home_lon = rng.uniform(-123.0, -71.0, n_cards)
    base_amt = rng.uniform(2.0, 4.0, n_cards)  # log-amount mean
    cards = [f"{rng.integers(4000, 4999)}{i:08d}" for i in range(n_cards)]

    secs = days * 24 * 3600
    recs = []
    for k in range(rows):
        c = int(rng.integers(0, n_cards))
        ts = ORIGIN + pd.Timedelta(seconds=int(rng.integers(0, secs)))
        fraud = rng.random() < 0.020
        if fraud:
            amount = float(np.round(np.exp(rng.normal(base_amt[c] + 1.8, 0.6)), 2))
            # far from home, often odd hours
            lat = float(np.round(home_lat[c] + rng.normal(0, 6.0), 4))
            lon = float(np.round(home_lon[c] + rng.normal(0, 6.0), 4))
            hour = int(rng.choice(range(24), p=_odd_hours()))
            category = str(rng.choice(["online", "electronics", "cash_advance", "travel"]))
        else:
            amount = float(np.round(np.exp(rng.normal(base_amt[c], 0.5)), 2))
            lat = float(np.round(home_lat[c] + rng.normal(0, 0.4), 4))
            lon = float(np.round(home_lon[c] + rng.normal(0, 0.4), 4))
            hour = int(rng.choice(range(24), p=_day_hours()))
            category = str(rng.choice(CATEGORIES))
        ts = ts.normalize() + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
        recs.append(
            (
                c,
                cards[c],
                ts,
                amount,
                f"m_{rng.integers(0, 800):03d}",
                category,
                lat,
                lon,
                int(fraud),
            )
        )

    df = pd.DataFrame(
        recs,
        columns=[
            "_card_idx",
            "cc_num",
            "datetime",
            "amount",
            "merchant",
            "category",
            "lat",
            "long",
            "is_fraud",
        ],
    )
    df = df.sort_values("datetime").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"T{i:09d}" for i in range(len(df))])
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df.drop(columns=["_card_idx"])


def _odd_hours() -> np.ndarray:
    w = np.ones(24)
    w[0:5] = 5.0
    return w / w.sum()  # fraud skews to night


def _day_hours() -> np.ndarray:
    w = np.ones(24)
    w[8:21] = 4.0
    return w / w.sum()  # legit skews to daytime


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", type=int, default=300)
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--rows", type=int, default=40000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    df = build(args.cards, args.days, args.rows)
    df.to_csv(args.out, index=False)
    print(
        f"[build_ccfraud] wrote {len(df)} rows ({df['is_fraud'].mean():.3%} fraud) "
        f"over {df['cc_num'].nunique()} cards -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
