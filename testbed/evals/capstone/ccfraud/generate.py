"""Capstone: credit-card fraud (classification) — generator.

World: a committed synthetic transaction log (built keyless by
`_data/build_ccfraud.py`). Per instance we sample a seeded subset of cards and
split chronologically: the earlier `data/transactions.csv` (with `is_fraud`) is
the training slice; the later `data/score_transactions.csv` (features only) is
the scoring slice whose labels are hidden in `solution/`.

The agent must build the FTI pipeline ON the platform — engineer features
(velocity / geo / amount) into a feature group, assemble a training dataset,
train + register a classifier, and write a per-transaction fraud probability to
the predictions feature table. Graded hybrid (metric AUC + artifacts) by
`evals.capstone.common.grade_capstone`.

The pass bar is calibrated per instance from a naive base-rate baseline and a
deterministic reference GradientBoosting model fit at generation time.

    python -m evals.capstone.ccfraud.generate --seed 7 --out /tmp/cc-7
    python -m evals.capstone.ccfraud.generate --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from evals.capstone import common
from evals.common import instance_suffix

METRIC = "auc"
TEST_FRAC = 0.18
N_CARDS = 220        # sampled per instance (of the fixture's 300)
RAW = common.RAW_DIR / "ccfraud_raw.csv"


class GateError(RuntimeError):
    pass


def _reference_features(df: pd.DataFrame, home: pd.DataFrame) -> pd.DataFrame:
    """Causal feature engineering used ONLY to calibrate the bar (mirrors what a
    competent agent would build): amount, hour, geo distance from the card's
    home, and 24h transaction velocity. Rolling windows look only at the past."""
    d = df.copy()
    d["dt"] = pd.to_datetime(d["datetime"], utc=True)
    d = d.merge(home, on="cc_num", how="left")
    d["amt_log"] = np.log1p(d["amount"])
    d["hour"] = d["dt"].dt.hour
    d["geo_dist"] = np.hypot(d["lat"] - d["home_lat"], d["long"] - d["home_lon"])
    parts = []  # per-card causal velocity, keyed by the unique transaction_id
    for _, g in d.groupby("cc_num", sort=False):
        gi = g.sort_values("dt").set_index("dt")
        parts.append(pd.DataFrame({
            "transaction_id": gi["transaction_id"].to_numpy(),
            "vel_cnt_24h": (gi["amount"].rolling("1D").count() - 1).to_numpy(),
            "vel_sum_24h": (gi["amount"].rolling("1D").sum() - gi["amount"]).to_numpy(),
            "secs_since_prev": gi.index.to_series().diff().dt.total_seconds()
            .fillna(1e6).to_numpy()}))
    vel = pd.concat(parts).set_index("transaction_id")
    feats = ["amt_log", "hour", "geo_dist"]
    return d.set_index("transaction_id")[feats].join(vel).fillna(0.0)


def generate(seed: int, out: Path) -> dict:
    if not RAW.exists():
        raise GateError(f"missing fixture {RAW}; run `python -m evals.capstone."
                        f"_data.build_ccfraud` once to build it")
    rng = np.random.default_rng(seed)
    sfx = instance_suffix(seed)
    raw = pd.read_csv(RAW)

    cards = np.sort(raw["cc_num"].unique())
    pick = rng.choice(cards, size=min(N_CARDS, len(cards)), replace=False)
    df = raw[raw["cc_num"].isin(pick)].sort_values("datetime").reset_index(drop=True)
    train, score = common.time_split(df, "datetime", TEST_FRAC)

    # --- reference bar ---------------------------------------------------------
    home = (train.groupby("cc_num")[["lat", "long"]].median()
            .rename(columns={"lat": "home_lat", "long": "home_lon"}).reset_index())
    feats = _reference_features(df, home)
    Xtr, ytr = feats.loc[train["transaction_id"]], train["is_fraud"].to_numpy()
    Xte, yte = feats.loc[score["transaction_id"]], score["is_fraud"].to_numpy()
    if yte.sum() == 0 or yte.sum() == len(yte):
        raise GateError(f"scoring slice is single-class (seed={seed})")
    clf = GradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
    ref_pred = clf.predict_proba(Xte)[:, 1]
    naive_pred = np.full(len(yte), ytr.mean())            # constant base rate
    cal = common.calibrate_bar(METRIC, yte, naive_pred, ref_pred, floor=0.70, margin=0.5)
    if cal["reference"] <= cal["naive"] + 0.05:
        raise GateError(f"reference model has no edge (seed={seed}): {cal}")

    # --- resource names + frames ----------------------------------------------
    names = {"feature_group": f"cctxn{sfx}", "training_dataset": f"cctd{sfx}",
             "model_name": f"ccmodel{sfx}", "predictions_table": f"ccpred{sfx}"}
    drop = ["is_fraud"]
    train_csv = train.drop(columns=[c for c in drop if c in train.columns]).assign(
        is_fraud=train["is_fraud"])
    score_csv = score.drop(columns=["is_fraud"])
    labels = score[["transaction_id", "is_fraud"]].copy()

    meta = {"family": "ccfraud", "seed": seed, "kind": "classification",
            "metric": METRIC, "id_col": "transaction_id",
            "pred_col": "fraud_probability", "label_col": "is_fraud",
            **cal, **names,
            "feature_group_version": 1, "training_dataset_version": 1,
            "predictions_version": 1,
            "record_ids": labels["transaction_id"].tolist(),
            "n_train": len(train), "n_test": len(score),
            "fraud_rate": round(float(train["is_fraud"].mean()), 4)}

    task_md = _task_md(names, cal)
    prompt = _prompt(names, cal)
    return common.write_instance(
        out, train=train_csv, score=score_csv, labels=labels,
        task_md=task_md, prompt=prompt, meta=meta,
        data_files={"transactions.csv": train_csv, "score_transactions.csv": score_csv})


def _task_md(n: dict, cal: dict) -> str:
    return (
        "# Capstone — credit-card fraud detection (classification)\n\n"
        "`data/transactions.csv` is a labelled history of card transactions "
        "(`transaction_id, cc_num, datetime, amount, merchant, category, lat, "
        "long, is_fraud`). `data/score_transactions.csv` holds later transactions "
        "WITHOUT the label — you must predict each one's fraud probability.\n\n"
        "Build the full pipeline on the platform:\n"
        f"1. Engineer fraud features (e.g. transaction velocity per card, geo "
        f"distance from the card's usual location, amount/hour signals) into a "
        f"feature group `{n['feature_group']}`.\n"
        f"2. Assemble a training dataset `{n['training_dataset']}` from it.\n"
        f"3. Train a fraud classifier and register it as `{n['model_name']}` "
        f"WITH its evaluation metrics.\n"
        f"4. Score every row of `score_transactions.csv` and write the results to "
        f"a feature table `{n['predictions_table']}` (record key "
        f"`transaction_id`, column `fraud_probability` in [0,1]).\n\n"
        f"Target: ROC AUC ≥ {cal['bar']:.3f} on the held-out scoring slice "
        f"(naive base-rate ≈ {cal['naive']:.3f}).\n"
    )


def _prompt(n: dict, cal: dict) -> str:
    return (
        "data/ has a labelled transaction history (data/transactions.csv, column "
        "`is_fraud`) and a later set of transactions to score with the label "
        "REMOVED (data/score_transactions.csv). See data/task.md.\n"
        "Build the full FTI pipeline ON THE PLATFORM: engineer fraud features "
        f"into feature group `{n['feature_group']}`, assemble training dataset "
        f"`{n['training_dataset']}`, train and register a classifier "
        f"`{n['model_name']}` (include its metrics), then score every row of "
        f"score_transactions.csv into a feature table `{n['predictions_table']}` "
        "with record key `transaction_id` and a `fraud_probability` column in "
        "[0,1]. Make the predictions table available for low-latency lookup as "
        "well, where the platform distinguishes online/offline.\n"
        f"You pass when held-out ROC AUC ≥ {cal['bar']:.3f}.\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            m = generate(seed, Path(f"/tmp/mlpab-ccfraud-selftest/{seed}"))
            print(f"[ccfraud] seed={seed} n_test={m['n_test']} bar={m['bar']} "
                  f"naive={m['naive']} ref={m['reference']} gates=OK")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
