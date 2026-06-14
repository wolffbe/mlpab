"""Drift task (FTI sub-category: ops/drift) — generator.

Usage:
    python -m evals.ops.drift.generate --seed 7 --out /tmp/drift-7
    python -m evals.ops.drift.generate --selftest

World: a daily feature stream (`data/features.csv` — one row per entity per
day, numeric features f1..f6). Exactly ONE feature's distribution shifts at a
seeded onset day (step change of ~4 baseline sigmas). The agent must identify
which feature drifted and when, and write `submission/answers.json`:

    {"feature": "f3", "onset": "2026-03-02"}

Ground truth by construction (the generator injected the drift). Gates: the
committed reference detector recovers (feature, onset±TOLERANCE) from the
emitted CSV; no other feature triggers it (uniqueness), and every wrong-feature
answer fails the grader.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ORIGIN = pd.Timestamp("2026-01-01", tz="UTC")
N_DAYS = 90
N_ENTITIES = 150
FEATURES = [f"f{i}" for i in range(1, 7)]
ONSET_WINDOW = (45, 70)  # drift starts in this day range
SHIFT_SIGMAS = 4.0  # step size in units of the feature's sigma
TOLERANCE_DAYS = 3  # |reported onset - true onset| accepted


def _detect(df: pd.DataFrame) -> tuple[str, pd.Timestamp] | None:
    """Committed reference detector: per feature, compare each day's mean to
    the first-30-day baseline; the first day where it deviates by more than
    6 baseline standard errors starts the drift. Returns (feature, onset) of
    the single drifted feature, or None if zero/multiple features trigger."""
    df = df.copy()
    df["day"] = pd.to_datetime(df["event_time"], utc=True).dt.floor("D")
    daily = df.groupby("day")[FEATURES].mean()
    base = daily.iloc[:30]
    hits: list[tuple[str, pd.Timestamp]] = []
    for f in FEATURES:
        mu, se = base[f].mean(), base[f].std()
        dev = (daily[f] - mu).abs() > 6 * se
        dev = dev[dev.index >= daily.index[30]]
        if dev.any():
            first = dev.idxmax()  # first True
            # require persistence (not a one-day fluke)
            after = dev.loc[first:]
            if after.mean() > 0.8:
                hits.append((f, first))
    if len(hits) != 1:
        return None
    return hits[0]


class GateError(RuntimeError):
    pass


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    drifted = str(rng.choice(FEATURES))
    onset_day = int(rng.integers(*ONSET_WINDOW))
    onset = ORIGIN + pd.Timedelta(days=onset_day)

    mus = rng.uniform(0, 10, len(FEATURES))
    sigmas = rng.uniform(0.5, 2.0, len(FEATURES))

    rows = []
    for d in range(N_DAYS):
        ts = ORIGIN + pd.Timedelta(days=d)
        vals = rng.normal(mus, sigmas, size=(N_ENTITIES, len(FEATURES)))
        if d >= onset_day:
            k = FEATURES.index(drifted)
            vals[:, k] += SHIFT_SIGMAS * sigmas[k]
        for e in range(N_ENTITIES):
            rows.append([f"E{e:04d}", ts.strftime("%Y-%m-%dT%H:%M:%SZ"), *np.round(vals[e], 4)])
    df = pd.DataFrame(rows, columns=["entity_id", "event_time", *FEATURES])

    # --- gates ---------------------------------------------------------------
    got = _detect(df)
    if got is None:
        raise GateError(f"reference detector found zero/multiple drifts (seed={seed})")
    got_f, got_onset = got
    if got_f != drifted or abs((got_onset - onset).days) > TOLERANCE_DAYS:
        raise GateError(
            f"reference detector disagrees with seeded truth "
            f"({got_f}@{got_onset.date()} vs {drifted}@{onset.date()}, seed={seed})"
        )

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    df.to_csv(out / "data" / "features.csv", index=False)
    (out / "data" / "schema.md").write_text(
        "# Schema\n\n- **features.csv**: entity_id, event_time (daily), "
        + ", ".join(FEATURES)
        + " (numeric features)\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains a daily stream of feature observations "
        "(data/features.csv: entity_id, event_time, "
        f"{', '.join(FEATURES)}).\n"
        "Exactly ONE of these features changed its distribution at some point "
        "in the stream (data drift). Use the platform's statistics/monitoring "
        "capabilities to investigate.\n"
        "Identify which feature drifted and the onset date, and write your "
        "answer to submission/answers.json as:\n"
        '    {"feature": "<name>", "onset": "YYYY-MM-DD"}\n'
    )
    truth = {
        "family": "drift",
        "seed": seed,
        "feature": drifted,
        "onset": onset.strftime("%Y-%m-%d"),
        "tolerance_days": TOLERANCE_DAYS,
    }
    (out / "solution" / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "instance.json").write_text(json.dumps({"family": "drift", "seed": seed}, indent=2))
    return truth


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-drift-selftest/{seed}"))
            print(f"[drift] seed={seed} gates=OK truth={meta['feature']}@{meta['onset']}")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
