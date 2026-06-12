"""Prediction-monitoring task (FTI sub-category: ops/prediction_monitoring) — generator.

Usage:
    python -m evals.ops.prediction_monitoring.generate --seed 7 --out /tmp/predmon-7
    python -m evals.ops.prediction_monitoring.generate --selftest

World: a deployed model's prediction log (`data/prediction_log.csv` — ts,
prediction; ~150 predictions per day over 90 days). The PREDICTION
distribution undergoes a step change (mean shift of ~4 baseline sigmas) at a
seeded onset day. The agent must log/monitor the predictions on the platform,
identify WHEN the distribution shifted, and write `submission/answers.json`:

    {"onset": "2026-03-02"}

Ground truth by construction (the generator injected the shift). Gates: the
committed reference detector recovers the onset within tolerance from the
emitted CSV, and a wrong onset fails the grade function. Detection logic is
self-contained (no imports from evals.ops.drift).
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
PER_DAY = 150                  # ~150 predictions/day (±20 jitter)
ONSET_WINDOW = (45, 70)        # the shift starts in this day range
SHIFT_SIGMAS = 4.0             # step size in units of the prediction sigma
TOLERANCE_DAYS = 3             # |reported onset - true onset| accepted


def _detect(df: pd.DataFrame) -> pd.Timestamp | None:
    """Committed reference detector: compare each day's mean prediction to the
    first-30-day baseline; the first day deviating by more than 6 baseline
    standard deviations (of the daily means), with persistence, is the onset."""
    df = df.copy()
    df["day"] = pd.to_datetime(df["ts"], utc=True).dt.floor("D")
    daily = df.groupby("day")["prediction"].mean()
    base = daily.iloc[:30]
    mu, s = base.mean(), base.std()
    dev = (daily - mu).abs() > 6 * s
    dev = dev[dev.index >= daily.index[30]]
    if not dev.any():
        return None
    first = dev.idxmax()
    if dev.loc[first:].mean() <= 0.8:      # require persistence, not a fluke
        return None
    return first


class GateError(RuntimeError):
    pass


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    onset_day = int(rng.integers(*ONSET_WINDOW))
    onset = ORIGIN + pd.Timedelta(days=onset_day)
    mu = float(rng.uniform(2, 8))
    sigma = float(rng.uniform(0.3, 1.0))

    rows = []
    for d in range(N_DAYS):
        day = ORIGIN + pd.Timedelta(days=d)
        n = PER_DAY + int(rng.integers(-20, 21))
        vals = rng.normal(mu + (SHIFT_SIGMAS * sigma if d >= onset_day else 0.0),
                          sigma, n)
        secs = np.sort(rng.integers(0, 86400, n))
        for sec, v in zip(secs, vals):
            rows.append([(day + pd.Timedelta(seconds=int(sec))).strftime(
                "%Y-%m-%dT%H:%M:%SZ"), round(float(v), 4)])
    df = pd.DataFrame(rows, columns=["ts", "prediction"])

    # --- gates ---------------------------------------------------------------
    got = _detect(df)
    if got is None:
        raise GateError(f"reference detector found no shift (seed={seed})")
    if abs((got - onset).days) > TOLERANCE_DAYS:
        raise GateError(f"reference detector disagrees with seeded onset "
                        f"({got.date()} vs {onset.date()}, seed={seed})")

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    df.to_csv(out / "data" / "prediction_log.csv", index=False)
    (out / "data" / "schema.md").write_text(
        "# Schema\n\n- **prediction_log.csv**: ts (ISO UTC), prediction "
        "(the deployed model's output, float)\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains a deployed model's logged predictions "
        "(data/prediction_log.csv: ts, prediction — about 150 predictions per "
        "day over 90 days).\n"
        "Use the platform to log/monitor these predictions (its prediction "
        "logging / model monitoring / statistics capabilities, as available).\n"
        "The prediction distribution SHIFTED at some point in the log. "
        "Identify when, and write your answer to submission/answers.json as:\n"
        '    {"onset": "YYYY-MM-DD"}\n'
    )
    truth = {
        "family": "prediction_monitoring", "seed": seed,
        "onset": onset.strftime("%Y-%m-%d"),
        "tolerance_days": TOLERANCE_DAYS,
    }
    (out / "solution" / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "instance.json").write_text(json.dumps(
        {"family": "prediction_monitoring", "seed": seed}, indent=2))

    # gate the grade function: detected onset passes, a far-off onset fails
    from evals.ops.prediction_monitoring.grade import grade
    if not grade(out, {"onset": got.strftime("%Y-%m-%d")})["success"]:
        raise GateError("reference answers fail the grade function")
    bad = (onset + pd.Timedelta(days=TOLERANCE_DAYS + 5)).strftime("%Y-%m-%d")
    if grade(out, {"onset": bad})["success"]:
        raise GateError("out-of-tolerance onset passes the grade function")
    return truth


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/banter-predmon-selftest/{seed}"))
            print(f"[prediction_monitoring] seed={seed} gates=OK truth={meta['onset']}")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
