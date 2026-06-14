"""Model-registration task (FTI sub-category: training/register) — generator.

Usage:
    python -m evals.training.register.generate --seed 7 --out /tmp/register-7
    python -m evals.training.register.generate --selftest

The world provides a small deterministic model artifact (data/model.json:
weights + bias + feature names) and its evaluation metrics
(data/metrics.json: auc, accuracy). The agent must register the model in the
platform's model registry as `churnmodel<sfx>` (per-instance suffix) version 1,
attaching the provided
metrics, and report what it registered in submission/answers.json.

Gates (no platform needed — the grade fn is called directly with adapter
`none`): the reference answers dict passes; wrong metrics fail; a wrong model
name fails.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from evals.common import instance_suffix
from evals.training.register.grade import grade_answers

MODEL_BASE = "churnmodel"  # per-instance: f"{MODEL_BASE}{instance_suffix(seed)}"
VERSION = 1
FEATURES = ["f1", "f2", "f3", "f4", "f5"]


class GateError(RuntimeError):
    pass


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    model_name = MODEL_BASE + instance_suffix(seed)
    model = {
        "model_type": "logistic_regression",
        "features": FEATURES,
        "weights": [round(float(w), 6) for w in rng.uniform(-2.0, 2.0, len(FEATURES))],
        "bias": round(float(rng.uniform(-0.5, 0.5)), 6),
    }
    metrics = {
        "auc": round(float(rng.uniform(0.80, 0.95)), 4),
        "accuracy": round(float(rng.uniform(0.70, 0.90)), 4),
    }

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    (out / "data" / "model.json").write_text(json.dumps(model, indent=2))
    (out / "data" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "prompt.txt").write_text(
        "The directory data/ contains a trained model artifact (data/model.json: "
        "weights, bias and feature names) and its evaluation metrics "
        "(data/metrics.json).\n"
        f"Register this model in the platform's model registry as `{model_name}`, "
        f"version {VERSION}, attaching the provided metrics from data/metrics.json "
        "to the registry entry, with the artifact file as the model's content.\n"
        "Finally write submission/answers.json as:\n"
        f'    {{"model_name": "{model_name}", "version": {VERSION}, '
        '"metrics": <the metrics you attached>}\n'
    )
    truth = {
        "family": "register",
        "seed": seed,
        "model_name": model_name,
        "version": VERSION,
        "metrics": metrics,
    }
    (out / "solution" / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "instance.json").write_text(json.dumps({"family": "register", "seed": seed}, indent=2))

    # --- gates (call the grade fn directly — no platform needed) ----------------
    reference = {
        "model_name": model_name,
        "version": VERSION,
        "metrics": json.loads((out / "data" / "metrics.json").read_text()),
    }
    if not grade_answers(out, "none", reference)["success"]:
        raise GateError(f"reference answers fail the grade fn (seed={seed})")
    wrong_metrics = {**reference, "metrics": {k: round(v + 0.01, 4) for k, v in metrics.items()}}
    if grade_answers(out, "none", wrong_metrics)["success"]:
        raise GateError(f"wrong-metrics answers pass the grade fn (seed={seed})")
    wrong_name = {**reference, "model_name": f"{model_name}final"}
    if grade_answers(out, "none", wrong_name)["success"]:
        raise GateError(f"wrong-name answers pass the grade fn (seed={seed})")

    return truth


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-register-selftest/{seed}"))
            print(f"[register] seed={seed} metrics={meta['metrics']} gates=OK")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
