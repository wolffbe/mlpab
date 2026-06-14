"""Leakage-detection task (feature/leakage) — generator.

Usage:
    python -m evals.feature.leakage.generate --seed 7 --out /tmp/leakage-7
    python -m evals.feature.leakage.generate --selftest

An existing training dataset (data/training_data.csv: features f1..f6 +
binary label) contains exactly ONE leaky feature — it was computed from
post-outcome data, so it predicts the label near-perfectly while every honest
feature carries only a weak signal. The agent must identify it and write
`submission/answers.json`:

    {"feature": "f4"}

Gates: the committed reference detector (per-feature separation between label
classes) finds the seeded feature uniquely; wrong-feature answers fail.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_ROWS = 800
FEATURES = [f"f{i}" for i in range(1, 7)]


def _detect(df: pd.DataFrame) -> str | None:
    """Committed reference detector: rank features by |standardized mean
    difference| between label classes; the leak dominates by a wide margin."""
    scores = {}
    for f in FEATURES:
        a, b = df.loc[df["label"] == 1, f], df.loc[df["label"] == 0, f]
        pooled = df[f].std() or 1.0
        scores[f] = abs(a.mean() - b.mean()) / pooled
    ranked = sorted(scores, key=scores.get, reverse=True)
    top, second = scores[ranked[0]], scores[ranked[1]]
    return ranked[0] if top > 3 * max(second, 1e-9) else None


class GateError(RuntimeError):
    pass


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    leaky = str(rng.choice(FEATURES))
    label = (rng.uniform(size=N_ROWS) < 0.35).astype(int)

    df = pd.DataFrame({"row_id": [f"R{i:05d}" for i in range(N_ROWS)]})
    for f in FEATURES:
        base = rng.normal(0, 1, N_ROWS)
        if f == leaky:
            # post-outcome computation: almost a re-encoding of the label
            df[f] = np.round(label * 2.5 + rng.normal(0, 0.15, N_ROWS), 6)
        else:
            # honest weak signal
            df[f] = np.round(base + label * rng.uniform(0.05, 0.25), 6)
    df["label"] = label

    got = _detect(df)
    if got != leaky:
        raise GateError(f"reference detector found {got!r}, seeded {leaky!r} (seed={seed})")

    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    df.to_csv(out / "data" / "training_data.csv", index=False)
    (out / "data" / "schema.md").write_text(
        "# Schema\n\n- **training_data.csv**: row_id, "
        + ", ".join(FEATURES)
        + " (numeric features), label (0/1 — the prediction target)\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains an existing training dataset "
        "(data/training_data.csv: features "
        f"{', '.join(FEATURES)} and the binary `label`).\n"
        "Exactly ONE of the features leaks the outcome: it was computed from "
        "post-outcome data, so it predicts the label far too well to be a real "
        "feature. Identify it and write your answer to submission/answers.json as:\n"
        '    {"feature": "<name>"}\n'
        'You may add a free-text "evidence" key describing what you found '
        "(optional).\n"
    )
    truth = {"family": "leakage", "seed": seed, "feature": leaky}
    (out / "solution" / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "instance.json").write_text(json.dumps({"family": "leakage", "seed": seed}, indent=2))
    return truth


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-leakage-selftest/{seed}"))
            print(f"[leakage] seed={seed} gates=OK truth={meta['feature']}")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
