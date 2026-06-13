"""Shared scaffolding for capstone FTI evals.

A capstone gives the agent a raw training slice (with labels) and a scoring
slice (features only); the agent must build the full FTI pipeline ON the
platform — engineer features into a feature group, assemble a training dataset,
train + register a model, and write per-row predictions to a feature table —
then we grade HYBRID:

  * metric   — the held-out predictive metric (AUC ↑ for classification,
               RMSE ↓ for regression) must clear a per-instance `bar`,
               calibrated between a naive baseline and a competent reference
               model fit at generation time (sklearn, deterministic).
  * artifacts — on a real platform (`--adapter != none`) the feature group,
               training dataset, and registered model must exist, read back
               through the adapter's state checks.

Determinism: the split, the naive baseline, and the reference-model fit are all
seeded, so the bar embedded in truth.json reproduces exactly. The GRADER itself
is numpy/pandas-only (it just scores the agent's predictions) — no model
training at grade time.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent / "_data"


# --- metrics (numpy/pandas only) ------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based ROC AUC (tie-corrected). NaN if a class is absent."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# Per-metric comparison: (better-direction predicate, human arrow).
METRICS: dict[str, dict[str, Any]] = {
    "auc": {"fn": roc_auc, "passes": lambda got, bar: got >= bar, "arrow": "↑"},
    "rmse": {"fn": rmse, "passes": lambda got, bar: got <= bar, "arrow": "↓"},
    "mae": {"fn": mae, "passes": lambda got, bar: got <= bar, "arrow": "↓"},
}


# --- seeded time split ----------------------------------------------------------
def time_split(df: pd.DataFrame, time_col: str, test_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: the latest `test_frac` of rows by `time_col` become
    the scoring slice (no leakage — train is strictly earlier)."""
    df = df.sort_values(time_col).reset_index(drop=True)
    cut = int(round(len(df) * (1.0 - test_frac)))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


# --- instance writing -----------------------------------------------------------
def write_instance(out: Path, *, train: pd.DataFrame, score: pd.DataFrame,
                   labels: pd.DataFrame, task_md: str, prompt: str,
                   meta: dict, data_files: dict[str, pd.DataFrame]) -> dict:
    """Materialize the instance: data/ (agent-visible), solution/ (answer key)."""
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    for name, frame in data_files.items():
        frame.to_csv(out / "data" / name, index=False)
    (out / "data" / "task.md").write_text(task_md)
    labels.to_csv(out / "solution" / "test_labels.csv", index=False)
    (out / "solution" / "truth.json").write_text(json.dumps(meta, indent=2, default=str))
    (out / "prompt.txt").write_text(prompt)
    (out / "instance.json").write_text(
        json.dumps({"family": meta["family"], "seed": meta["seed"]}, indent=2))
    return meta


# --- hybrid grading -------------------------------------------------------------
def _read_predictions(adapter: str | None, table: str, version: int,
                      record_ids: list[str] | None, run_dir: Path) -> pd.DataFrame:
    """Predictions read back THROUGH the platform (or local CSV for `none`)."""
    if adapter in ("hopsworks", "databricks", "sagemaker"):
        from evals.common import fetch_table
        return fetch_table(adapter, table, version, record_ids)
    local = run_dir / "submission" / f"{table}.csv"
    if not local.exists():
        raise LookupError(f"no local predictions at {local}")
    return pd.read_csv(local)


def grade_capstone(instance_dir: Path, adapter: str, run_dir: Path) -> dict:
    """Hybrid grade shared by every capstone (truth.json carries all specifics).

    asserts: A1 predictions exist & cover the test rows · A2 metric clears the
    bar · (real platform only) A3 feature group · A4 training dataset · A5 model
    registered with metrics.
    """
    truth = json.loads((instance_dir / "solution" / "truth.json").read_text())
    labels = pd.read_csv(instance_dir / "solution" / "test_labels.csv")
    id_col, pred_col, label_col = truth["id_col"], truth["pred_col"], truth["label_col"]
    mspec = METRICS[truth["metric"]]
    asserts: list[dict] = []
    diagnostic: str | None = None

    def check(name, ok, detail=""):
        asserts.append({"name": name, "passed": bool(ok),
                        **({"detail": detail} if detail and not ok else {})})
        return bool(ok)

    # A1 — predictions table exists, has the right columns, covers every row ----
    preds = None
    try:
        preds = _read_predictions(adapter, truth["predictions_table"],
                                  truth.get("predictions_version", 1),
                                  truth.get("record_ids"), run_dir)
        preds.columns = [str(c).strip().lower() for c in preds.columns]
    except (LookupError, NotImplementedError) as e:
        check("A1_predictions_table", False, str(e))

    complete = False
    if preds is not None:
        have_cols = id_col in preds.columns and pred_col in preds.columns
        check("A1_predictions_table", have_cols,
              f"need columns [{id_col}, {pred_col}], got {list(preds.columns)}")
        if have_cols:
            # Coerce both id columns to str before merging: a platform readback
            # (BigQuery/Hopsworks) may hand back a date/timestamp id as datetime
            # while the labels carry it as a string, which makes pandas raise
            # "merge on object and datetime64 columns" — crashing the grader
            # instead of failing A1 gracefully. keep="last" so an agent's most
            # recent overwrite for an id wins over a stale earlier row.
            labels[id_col] = labels[id_col].astype(str)
            preds[id_col] = preds[id_col].astype(str)
            merged = labels.merge(
                preds[[id_col, pred_col]].drop_duplicates(id_col, keep="last"),
                on=id_col, how="left")
            n_missing = int(merged[pred_col].isna().sum())
            complete = check("A1_coverage", n_missing == 0,
                             f"{n_missing}/{len(merged)} scored rows missing a prediction")
            # A2 — metric clears the bar --------------------------------------
            scored = merged.dropna(subset=[pred_col])
            if len(scored):
                got = mspec["fn"](scored[label_col].to_numpy(), scored[pred_col].to_numpy())
                bar = float(truth["bar"])
                passed = bool(complete and mspec["passes"](got, bar))
                check("A2_metric", passed,
                      f"{truth['metric']}={got:.4f} {mspec['arrow']} bar={bar:.4f} "
                      f"(naive={truth['naive']:.4f}, reference={truth['reference']:.4f})"
                      + ("" if complete else " — incomplete predictions"))
                if complete and not passed:
                    diagnostic = (f"model is no better than the naive baseline "
                                  f"({truth['metric']}={got:.4f} vs naive "
                                  f"{truth['naive']:.4f}); the FTI pipeline ran but the "
                                  f"model has no signal")

    # A3–A5 — on-platform FTI artifacts (real platforms only) -------------------
    if adapter != "none":
        from evals.common import state_checker
        ck = state_checker(adapter)
        fg = ck.get_feature_table(truth["feature_group"], truth.get("feature_group_version", 1))
        check("A3_feature_group", fg is not None,
              f"feature group {truth['feature_group']!r} not found on {adapter}")
        try:
            td = ck.read_training_dataset(truth["training_dataset"],
                                          truth.get("training_dataset_version", 1))
            check("A4_training_dataset", td is not None and len(td) > 0,
                  f"training dataset {truth['training_dataset']!r} empty/missing")
        except Exception as e:  # noqa: BLE001 — adapter raises platform-specific errors
            check("A4_training_dataset", False, f"training dataset read failed: {e}")
        model = ck.get_model(truth["model_name"])
        has_metrics = bool(model and model.get("exists") and model.get("metrics"))
        check("A5_model_registered", has_metrics,
              f"model {truth['model_name']!r} not registered with metrics on {adapter}")

    success = all(a["passed"] for a in asserts) and len(asserts) > 0
    return {"family": truth["family"], "seed": truth["seed"], "success": success,
            "asserts_passed": sum(a["passed"] for a in asserts),
            "asserts_total": len(asserts), "asserts": asserts,
            **({"diagnostic": diagnostic} if diagnostic and not success else {})}


def grade_main(family: str, argv: list[str] | None = None) -> int:
    """`platform`-kind CLI: --instance + --adapter <name|none>."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=Path, required=True)
    ap.add_argument("--adapter", required=True,
                    choices=["hopsworks", "databricks", "sagemaker", "none"])
    args = ap.parse_args(argv)
    report = grade_capstone(args.instance, args.adapter, Path.cwd())
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["success"] else 1


# --- reference-model bar calibration (generation time only) --------------------
def calibrate_bar(metric: str, y_true: np.ndarray, naive_pred: np.ndarray,
                  ref_pred: np.ndarray, *, floor: float, margin: float) -> dict:
    """Turn a naive baseline + a competent reference model into a pass `bar`.

    The bar sits a fraction `margin` of the way from the naive metric toward the
    reference metric, clamped by a domain `floor`, so a real model clears it and
    a trivial / broken pipeline does not. Returns {bar, naive, reference}.
    """
    fn = METRICS[metric]["fn"]
    naive = fn(y_true, naive_pred)
    reference = fn(y_true, ref_pred)
    if metric == "auc":                       # higher is better
        bar = naive + margin * (reference - naive)
        # Apply the domain floor, but never push the bar past the reference —
        # otherwise a model that matches the competent reference would fail an
        # unwinnable bar (floor can exceed a weak reference's AUC).
        bar = min(max(bar, floor), reference)
    else:                                     # rmse/mae — lower is better
        bar = reference + (1.0 - margin) * (naive - reference)
        # Floor caps how lenient the bar may be; never make it stricter than the
        # reference's own error, which would be unwinnable.
        bar = max(min(bar, floor), reference)
    return {"bar": round(float(bar), 4), "naive": round(float(naive), 4),
            "reference": round(float(reference), 4)}
