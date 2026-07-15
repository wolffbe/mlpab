"""Skew task (FTI sub-category: inference/skew) — generator.

Usage:
    python -m evals.inference.skew.generate --seed 7 --out /tmp/skew-7
    python -m evals.inference.skew.generate --selftest

World: the training-time feature matrix of a deployed model
(`data/training_sample.csv`) and the feature vectors the online service
actually served for the same entities (`data/serving_log.csv`). Exactly ONE
feature is computed differently between the two paths — the classic
training/serving skew (here: the training pipeline applied log1p, the serving
path forgot and uses raw values). All other features agree exactly per entity.

The agent must identify the skewed feature and write
`submission/answers.json`:

    {"feature": "f2"}

(an optional free-text "cause" key is welcome but ungraded). Gates: the
committed reference detector recovers the seeded feature from the emitted CSVs
and finds exactly one diverging feature; wrong-feature answers fail the grader.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KIND = "answers"  # deliverable kind: table | dataset | answers | platform
SUMMARY = (
    "Measures whether an agent can diagnose training/serving skew: spotting the single feature "
    "computed differently between the training pipeline and the online serving path."
)

N_TRAIN = 300
N_SERVED = 200
FEATURES = [f"f{i}" for i in range(1, 6)]


def _detect(train: pd.DataFrame, served: pd.DataFrame) -> str | None:
    """Committed reference detector: join on entity_id, find features whose
    per-entity values systematically differ. Returns the single diverging
    feature, or None if zero/multiple diverge."""
    j = train.merge(served, on="entity_id", suffixes=("_t", "_s"))
    diverging = [f for f in FEATURES if (j[f + "_t"] - j[f + "_s"]).abs().mean() > 1e-9]
    return diverging[0] if len(diverging) == 1 else None


class GateError(RuntimeError):
    pass


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    skewed = str(rng.choice(FEATURES))

    entities = [f"E{i:04d}" for i in range(N_TRAIN)]
    raw = {f: rng.lognormal(2.0, 0.8, N_TRAIN) for f in FEATURES}

    # Training path: every feature as the pipeline computed it — the skewed
    # feature was log1p-transformed there.
    train = pd.DataFrame({"entity_id": entities})
    for f in FEATURES:
        train[f] = np.round(np.log1p(raw[f]) if f == skewed else raw[f], 6)

    # Serving path: a subset of entities, shuffled; the skewed feature is
    # served RAW (the transform was forgotten), everything else matches.
    idx = rng.choice(N_TRAIN, N_SERVED, replace=False)
    served = pd.DataFrame({"entity_id": [entities[i] for i in idx]})
    for f in FEATURES:
        vals = raw[f][idx]
        served[f] = np.round(vals if f == skewed else vals, 6)
        if f != skewed:
            served[f] = train[f].to_numpy()[idx]  # exact agreement
    served = served.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # --- gates ---------------------------------------------------------------
    got = _detect(train, served)
    if got != skewed:
        raise GateError(f"reference detector found {got!r}, seeded {skewed!r} (seed={seed})")

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    train.to_csv(out / "data" / "training_sample.csv", index=False)
    served.to_csv(out / "data" / "serving_log.csv", index=False)
    (out / "data" / "schema.md").write_text(
        "# Schema\n\n- **training_sample.csv**: entity_id, "
        + ", ".join(FEATURES)
        + " — features as the TRAINING pipeline computed them\n"
        "- **serving_log.csv**: entity_id, "
        + ", ".join(FEATURES)
        + " — feature vectors the ONLINE service actually served\n"
    )
    (out / "prompt.txt").write_text(
        "A model was trained on the feature matrix in data/training_sample.csv. "
        "The online service logs the feature vectors it actually serves in "
        "data/serving_log.csv (same entities, same feature names).\n"
        "Exactly ONE feature is computed differently between the training path "
        "and the serving path (training/serving skew). Identify which feature "
        "diverges and write your answer to submission/answers.json as:\n"
        '    {"feature": "<name>"}\n'
        'You may add a free-text "cause" key describing what you think went '
        "wrong (optional).\n"
    )
    truth = {
        "family": "skew",
        "seed": seed,
        "feature": skewed,
        "cause": "training applied log1p; serving uses raw values",
    }
    (out / "solution" / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "instance.json").write_text(json.dumps({"family": "skew", "seed": seed}, indent=2))
    return truth


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-skew-selftest/{seed}"))
            print(f"[skew] seed={seed} gates=OK truth={meta['feature']}")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
