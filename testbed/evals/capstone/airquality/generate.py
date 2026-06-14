"""Capstone: air-quality PM2.5 forecasting (regression) — generator.

World: a committed daily PM2.5 + weather history (built by
`_data/build_airquality.py` from AQICN PM2.5 + Open-Meteo weather; a keyless
synthetic placeholder has the same schema). Per instance we split chronologically
— `data/airquality_history.csv` (with `pm25`) is the training history; a seeded
set of later non-adjacent days, `data/forecast_days.csv` (features + the prior
day's `pm25_lag1`, but NOT the day's `pm25`), is the scoring slice whose true
PM2.5 is hidden in `solution/`.

The agent must build the FTI pipeline ON the platform — engineer features into a
feature group, assemble a training dataset, train + register a regressor, and
write a predicted PM2.5 per day to the predictions feature table. Graded hybrid
(RMSE + artifacts) by `evals.capstone.common.grade_capstone`, with the bar
calibrated between a train-mean baseline and a deterministic reference model.

    python -m evals.capstone.airquality.generate --seed 7 --out /tmp/aq-7
    python -m evals.capstone.airquality.generate --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from evals.capstone import common
from evals.common import instance_suffix

METRIC = "rmse"
N_TEST = 90
REGION = 0.70  # test days are sampled from the last 30% of history
FEATURES = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]
RAW = common.RAW_DIR / "airquality_raw.csv"


class GateError(RuntimeError):
    pass


def _sample_test_idx(rng, lo: int, hi: int, n: int) -> list[int]:
    """Non-adjacent day indices in [lo, hi) — so each test day's `pm25_lag1`
    (the previous day) is never itself a hidden test day (no label leakage)."""
    cand = list(range(lo, hi))
    rng.shuffle(cand)
    chosen: list[int] = []
    used: set[int] = set()
    for i in cand:
        if (i - 1) in used or (i + 1) in used:
            continue
        chosen.append(i)
        used.add(i)
        if len(chosen) >= n:
            break
    return sorted(chosen)


def generate(seed: int, out: Path) -> dict:
    if not RAW.exists():
        raise GateError(
            f"missing fixture {RAW}; run `python -m evals.capstone."
            f"_data.build_airquality` once to build it"
        )
    rng = np.random.default_rng(seed)
    sfx = instance_suffix(seed)
    df = pd.read_csv(RAW).sort_values("date").reset_index(drop=True)
    df["pm25_lag1"] = df["pm25"].shift(1)
    df = df.dropna(subset=["pm25_lag1"]).reset_index(drop=True)

    n = len(df)
    region_start = int(n * REGION)
    test_idx = _sample_test_idx(rng, region_start, n, N_TEST)
    if len(test_idx) < N_TEST // 2:
        raise GateError(f"not enough non-adjacent test days (seed={seed})")
    train = df.iloc[:region_start].reset_index(drop=True)  # strictly earlier
    score = df.iloc[test_idx].reset_index(drop=True)

    # --- reference bar ---------------------------------------------------------
    Xtr, ytr = train[FEATURES], train["pm25"].to_numpy()
    Xte, yte = score[FEATURES], score["pm25"].to_numpy()
    reg = GradientBoostingRegressor(random_state=0).fit(Xtr, ytr)
    ref_pred = reg.predict(Xte)
    naive_pred = np.full(len(yte), ytr.mean())  # train-mean predictor
    cal = common.calibrate_bar(
        METRIC, yte, naive_pred, ref_pred, floor=common.rmse(yte, naive_pred), margin=0.5
    )
    if cal["reference"] >= 0.95 * cal["naive"]:
        raise GateError(f"reference model barely beats the mean (seed={seed}): {cal}")

    # --- resource names + frames ----------------------------------------------
    names = {
        "feature_group": f"airq{sfx}",
        "training_dataset": f"airqtd{sfx}",
        "model_name": f"airqmodel{sfx}",
        "predictions_table": f"airqpred{sfx}",
    }
    train_csv = train[["date", "pm25_lag1", *FEATURES[1:], "pm25"]]
    score_csv = score[["date", *FEATURES]]
    labels = score[["date", "pm25"]].copy()

    meta = {
        "family": "airquality",
        "seed": seed,
        "kind": "regression",
        "metric": METRIC,
        "id_col": "date",
        "pred_col": "pm25_pred",
        "label_col": "pm25",
        **cal,
        **names,
        "feature_group_version": 1,
        "training_dataset_version": 1,
        "predictions_version": 1,
        "record_ids": labels["date"].tolist(),
        "n_train": len(train),
        "n_test": len(score),
    }

    return common.write_instance(
        out,
        train=train_csv,
        score=score_csv,
        labels=labels,
        task_md=_task_md(names, cal),
        prompt=_prompt(names, cal),
        meta=meta,
        data_files={"airquality_history.csv": train_csv, "forecast_days.csv": score_csv},
    )


def _task_md(n: dict, cal: dict) -> str:
    return (
        "# Capstone — air-quality PM2.5 forecasting (regression)\n\n"
        "`data/airquality_history.csv` is a daily history (`date, pm25_lag1, "
        "temperature, humidity, wind_speed, pressure, precipitation, pm25`) where "
        "`pm25` is the measured air quality and `pm25_lag1` is the previous day's "
        "value. `data/forecast_days.csv` holds later days WITHOUT `pm25` — predict "
        "it for each.\n\n"
        "Build the full pipeline on the platform:\n"
        f"1. Engineer features (weather + lag/rolling air-quality signals) into a "
        f"feature group `{n['feature_group']}`.\n"
        f"2. Assemble a training dataset `{n['training_dataset']}`.\n"
        f"3. Train a PM2.5 regressor and register it as `{n['model_name']}` WITH "
        f"its evaluation metrics.\n"
        f"4. Predict every row of `forecast_days.csv` into a feature table "
        f"`{n['predictions_table']}` (record key `date`, column `pm25_pred`).\n\n"
        f"Target: RMSE ≤ {cal['bar']:.3f} µg/m³ on the held-out days "
        f"(train-mean baseline ≈ {cal['naive']:.3f}).\n"
    )


def _prompt(n: dict, cal: dict) -> str:
    return (
        "data/ has a daily air-quality + weather history (data/airquality_history.csv, "
        "target column `pm25`) and a later set of days to forecast with `pm25` "
        "REMOVED (data/forecast_days.csv). See data/task.md.\n"
        "Build the full FTI pipeline ON THE PLATFORM: engineer features into "
        f"feature group `{n['feature_group']}`, assemble training dataset "
        f"`{n['training_dataset']}`, train and register a regressor "
        f"`{n['model_name']}` (include its metrics), then predict every row of "
        f"forecast_days.csv into a feature table `{n['predictions_table']}` with "
        "record key `date` and a `pm25_pred` column. Make the predictions table "
        "available for low-latency lookup as well, where the platform "
        "distinguishes online/offline.\n"
        f"You pass when held-out RMSE ≤ {cal['bar']:.3f} µg/m³.\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            m = generate(seed, Path(f"/tmp/mlpab-airquality-selftest/{seed}"))
            print(
                f"[airquality] seed={seed} n_test={m['n_test']} bar={m['bar']} "
                f"naive={m['naive']} ref={m['reference']} gates=OK"
            )
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
