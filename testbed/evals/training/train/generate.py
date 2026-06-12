"""Training-job task (FTI sub-category: training/train) — generator.

Usage:
    python -m evals.training.train.generate --seed 7 --out /tmp/train-7
    python -m evals.training.train.generate --selftest

The world provides data/train.csv (row_id, f1..f5, label), data/score.csv
(row_id, f1..f5) and data/train_model.py — a fully deterministic pure
numpy+pandas training script (logistic regression by plain gradient descent:
learning rate 0.1, exactly 300 iterations, zero-initialized weights) that
writes predictions.csv (row_id, score rounded to 6 dp). The agent must run it
AS A JOB ON THE PLATFORM (named `trainjob<sfx>`), load the predictions into
feature table `predictions<sfx>` v1 (per-instance suffix), and report the job
name in answers.json.

Ground truth by construction: the generator reimplements the identical
training math inline; a gate additionally runs the emitted train_model.py via
subprocess in a temp dir and requires it to reproduce the truth digest
EXACTLY. Naive variant (gates assert it differs): training for only 30
iterations.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from evals.common import canonicalize, digest, instance_suffix

N_TRAIN = 500
N_SCORE = 200
FEATURES = ["f1", "f2", "f3", "f4", "f5"]
LEARNING_RATE = 0.1
ITERATIONS = 300
TABLE_BASE = "predictions"  # per-instance: f"{TABLE_BASE}{instance_suffix(seed)}"
JOB_BASE = "trainjob"       # per-instance: f"{JOB_BASE}{instance_suffix(seed)}"

SPEC = {
    "columns": ["row_id", "score"],
    "ts_cols": [],
    "int_cols": [],
    "float_cols": ["score"],
    "sort_cols": ["row_id"],
}
VARIANT_DIAGNOSIS = {
    "undertrained": "predictions come from an undertrained model "
                    "(30 iterations instead of the script's 300)",
}

TRAIN_SCRIPT = '''"""Provided training script — deterministic logistic regression.

Reads train.csv and score.csv from its working directory, trains by plain
gradient descent (learning rate 0.1, exactly 300 iterations, weights and bias
initialized to zero) and writes predictions.csv with columns row_id, score
(rounded to 6 decimals). Fully deterministic — do NOT modify it; run it as-is.
"""
import numpy as np
import pandas as pd

FEATURES = ["f1", "f2", "f3", "f4", "f5"]
LEARNING_RATE = 0.1
ITERATIONS = 300


def main():
    train = pd.read_csv("train.csv")
    score = pd.read_csv("score.csv")
    X = train[FEATURES].to_numpy(dtype=float)
    y = train["label"].to_numpy(dtype=float)
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    for _ in range(ITERATIONS):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = p - y
        w = w - LEARNING_RATE * (X.T @ g) / len(y)
        b = b - LEARNING_RATE * g.mean()
    Xs = score[FEATURES].to_numpy(dtype=float)
    preds = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
    out = pd.DataFrame({"row_id": score["row_id"], "score": np.round(preds, 6)})
    out.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
'''


class GateError(RuntimeError):
    pass


def _train_predict(train_df: pd.DataFrame, score_df: pd.DataFrame,
                   iterations: int) -> pd.DataFrame:
    """The identical training math, reimplemented inline (see TRAIN_SCRIPT)."""
    X = train_df[FEATURES].to_numpy(dtype=float)
    y = train_df["label"].to_numpy(dtype=float)
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = p - y
        w = w - LEARNING_RATE * (X.T @ g) / len(y)
        b = b - LEARNING_RATE * g.mean()
    Xs = score_df[FEATURES].to_numpy(dtype=float)
    preds = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
    return pd.DataFrame({"row_id": score_df["row_id"], "score": np.round(preds, 6)})


def _world(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    true_w = rng.uniform(-2.0, 2.0, len(FEATURES))
    true_b = float(rng.uniform(-0.5, 0.5))

    def make(n: int, with_label: bool, prefix: str) -> pd.DataFrame:
        X = np.round(rng.normal(0.0, 1.0, (n, len(FEATURES))), 6)
        df = pd.DataFrame(X, columns=FEATURES)
        df.insert(0, "row_id", [f"{prefix}{i:05d}" for i in range(n)])
        if with_label:
            p = 1.0 / (1.0 + np.exp(-(X @ true_w + true_b)))
            df["label"] = (rng.uniform(size=n) < p).astype(int)
        return df

    return make(N_TRAIN, True, "T"), make(N_SCORE, False, "S")


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    sfx = instance_suffix(seed)
    table = TABLE_BASE + sfx
    job_name = JOB_BASE + sfx
    train_df, score_df = _world(rng)

    truth = canonicalize(_train_predict(train_df, score_df, ITERATIONS), SPEC)
    variants = {
        "undertrained": canonicalize(_train_predict(train_df, score_df, 30), SPEC),
    }
    for name, v in variants.items():
        if digest(v) == digest(truth):
            raise GateError(f"variant {name!r} matches truth (seed={seed})")

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    train_df.to_csv(out / "data" / "train.csv", index=False)
    score_df.to_csv(out / "data" / "score.csv", index=False)
    (out / "data" / "train_model.py").write_text(TRAIN_SCRIPT)

    # Gate: the EMITTED script, run from the emitted data files in a clean temp
    # dir, must reproduce the truth digest exactly.
    with tempfile.TemporaryDirectory(prefix="banter-train-gate-") as td:
        for f in ("train.csv", "score.csv", "train_model.py"):
            shutil.copy(out / "data" / f, Path(td) / f)
        proc = subprocess.run([sys.executable, "train_model.py"], cwd=td,
                              capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise GateError(f"emitted train_model.py failed (seed={seed}): "
                            f"{proc.stderr[-500:]}")
        got = canonicalize(pd.read_csv(Path(td) / "predictions.csv"), SPEC)
        if digest(got) != digest(truth):
            raise GateError(f"emitted train_model.py does not reproduce the "
                            f"truth digest (seed={seed})")

    (out / "prompt.txt").write_text(
        "The directory data/ contains a training set (data/train.csv: row_id, "
        f"{', '.join(FEATURES)}, label), a scoring set (data/score.csv: row_id, "
        f"{', '.join(FEATURES)}) and a provided, fully deterministic training "
        "script (data/train_model.py) that reads both CSVs from its working "
        "directory and writes predictions.csv (row_id, score).\n"
        f"Run the provided script AS A JOB ON THE PLATFORM — named `{job_name}` — "
        "not locally on this machine. Do not modify the script or its "
        "hyperparameters.\n"
        f"Load the job's predictions output into a feature table named `{table}`, "
        "version 1, on the platform, with record key `row_id`.\n"
        "Make the table's features available for low-latency lookup as well "
        "(online/real-time access), where the platform distinguishes the two.\n"
        "Finally write submission/answers.json as:\n"
        f'    {{"job_name": "{job_name}"}}\n'
    )
    meta = {
        "family": "train", "seed": seed,
        "table_name": table, "table_version": 1,
        "job_name": job_name,
        "spec": SPEC, "row_count": len(truth), "digest": digest(truth),
        "record_ids": truth["row_id"].tolist(),
        "variant_digests": {k: digest(v) for k, v in variants.items()},
        "variant_diagnosis": VARIANT_DIAGNOSIS,
        "spot_rows": truth.head(3).to_dict(orient="records"),
    }
    (out / "solution" / "truth.json").write_text(json.dumps(meta, indent=2))
    (out / "instance.json").write_text(json.dumps({"family": "train", "seed": seed}, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/banter-train-selftest/{seed}"))
            print(f"[train] seed={seed} rows={meta['row_count']} gates=OK "
                  "(incl. subprocess reproduction)")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
