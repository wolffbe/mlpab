"""Model-dependent transformations task (FTI sub-category: training/mdt) — generator.

Usage:
    python -m evals.training.mdt.generate --seed 7 --out /tmp/mdt-7
    python -m evals.training.mdt.generate --selftest

Two splits of the same features (data/features_train.csv, 400 rows;
data/features_serve.csv, 250 rows; the generator seeds DIFFERENT
distributions per split) must land in one feature table `scaled<sfx>` v1
(per-instance suffix)
(row_id, split, f1..f4), standardized with mean/std fitted ON THE TRAIN SPLIT
ONLY: (x - train_mean) / train_std, population std (ddof=0), rounded to 6 dp.

Ground truth by construction. Naive variants (gates assert they differ):
fitting the statistics on both splits combined; standardizing each split with
its own statistics.
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

N_TRAIN = 400
N_SERVE = 250
FEATURES = ["f1", "f2", "f3", "f4"]
TABLE_BASE = "scaled"  # per-instance: f"{TABLE_BASE}{instance_suffix(seed)}"

SPEC = {
    "columns": ["row_id", "split", "f1", "f2", "f3", "f4"],
    "ts_cols": [],
    "int_cols": [],
    "float_cols": FEATURES,
    "sort_cols": ["row_id"],
}
VARIANT_DIAGNOSIS = {
    "fit_on_all": "scaling statistics were fitted on both splits combined "
                  "instead of the train split only",
    "per_split": "each split was standardized with its own statistics "
                 "instead of the train split's",
}


class GateError(RuntimeError):
    pass


def _standardize(df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for f in FEATURES:
        mu = stats_df[f].to_numpy(dtype=float).mean()
        sd = stats_df[f].to_numpy(dtype=float).std(ddof=0)
        out[f] = np.round((df[f].to_numpy(dtype=float) - mu) / sd, 6)
    return out


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    table = TABLE_BASE + instance_suffix(seed)

    def make(n: int, start: int, shift: float) -> pd.DataFrame:
        df = pd.DataFrame({"row_id": [f"R{i:05d}" for i in range(start, start + n)]})
        for f in FEATURES:
            mu = float(rng.uniform(-5.0, 5.0))
            sd = float(rng.uniform(0.5, 3.0))
            df[f] = np.round(rng.normal(mu + shift, sd, n), 6)
        return df

    train_df = make(N_TRAIN, 0, 0.0)
    # the serve split is drawn with shifted/rescaled distributions per feature,
    # so wrong fit populations provably change the result
    serve_df = make(N_SERVE, N_TRAIN, float(rng.uniform(1.5, 4.0)))

    both = pd.concat([train_df.assign(split="train"), serve_df.assign(split="serve")],
                     ignore_index=True)
    truth = canonicalize(
        pd.concat([_standardize(train_df, train_df).assign(split="train"),
                   _standardize(serve_df, train_df).assign(split="serve")],
                  ignore_index=True), SPEC)

    # --- gates ---------------------------------------------------------------
    variants = {
        "fit_on_all": canonicalize(
            _standardize(both, both[["row_id"] + FEATURES]), SPEC),
        "per_split": canonicalize(
            pd.concat([_standardize(train_df, train_df).assign(split="train"),
                       _standardize(serve_df, serve_df).assign(split="serve")],
                      ignore_index=True), SPEC),
    }
    for name, v in variants.items():
        if digest(v) == digest(truth):
            raise GateError(f"variant {name!r} matches truth (seed={seed})")
    # reference: independent pandas path — fit on train only, apply to both
    mu = train_df[FEATURES].mean()
    sd = train_df[FEATURES].std(ddof=0)
    ref = both.copy()
    ref[FEATURES] = ((both[FEATURES] - mu) / sd).round(6)
    if digest(canonicalize(ref, SPEC)) != digest(truth):
        raise GateError(f"reference standardization disagrees with truth (seed={seed})")

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    train_df.to_csv(out / "data" / "features_train.csv", index=False)
    serve_df.to_csv(out / "data" / "features_serve.csv", index=False)
    (out / "data" / "schema.md").write_text(
        "# Schema\n\nTwo splits of the same numeric features; `row_id` is unique "
        "across both files.\n\n- **features_train.csv**: row_id, "
        + ", ".join(FEATURES) + " — the training split\n"
        "- **features_serve.csv**: row_id, "
        + ", ".join(FEATURES) + " — the serving split\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains a training split (data/features_train.csv) "
        "and a serving split (data/features_serve.csv) of the same numeric "
        f"features ({', '.join(FEATURES)}); data/schema.md documents them.\n"
        f"Register a feature table named `{table}`, version 1, on the platform, "
        "with record key `row_id` and columns row_id, split, "
        f"{', '.join(FEATURES)}, containing BOTH splits standardized as a model "
        "preprocessing step: for each feature, (x - mean) / std, where the mean "
        "and the standard deviation over the training rows ONLY are used for both "
        "splits (the standard deviation is the population one, i.e. computed over "
        "the training rows without Bessel's correction). Round the standardized "
        "values to 6 decimals.\n"
        'The `split` column is "train" for rows from the training split and '
        '"serve" for rows from the serving split.\n'
        "Make the table's features available for low-latency lookup as well "
        "(online/real-time access), where the platform distinguishes the two.\n"
    )
    meta = {
        "family": "mdt", "seed": seed,
        "table_name": table, "table_version": 1,
        "spec": SPEC, "row_count": len(truth), "digest": digest(truth),
        "record_ids": truth["row_id"].tolist(),
        "variant_digests": {k: digest(v) for k, v in variants.items()},
        "variant_diagnosis": VARIANT_DIAGNOSIS,
        "spot_rows": truth.head(3).to_dict(orient="records"),
    }
    (out / "solution" / "truth.json").write_text(json.dumps(meta, indent=2))
    (out / "instance.json").write_text(json.dumps({"family": "mdt", "seed": seed}, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-mdt-selftest/{seed}"))
            print(f"[mdt] seed={seed} rows={meta['row_count']} gates=OK")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
