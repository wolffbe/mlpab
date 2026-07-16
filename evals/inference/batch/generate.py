"""Batch-scoring task (FTI sub-category: inference/batch) — generator.

Usage:
    python -m evals.inference.batch.generate --seed 7 --out /tmp/batch-7
    python -m evals.inference.batch.generate --selftest

World: a feature-history export (`data/feature_history.csv` — multiple
revisions per account over 60 days), the model (`data/model.json` — linear
weights + bias), and a scoring request (`data/scoring_request.md`) naming the
AS-OF timestamp T. The agent must batch-score EVERY account using the feature
values that were VALID AT time T (the most recent revision at or before T)
with score = sigmoid(w·x + b) rounded to 6 decimals, into a feature table
`scores<sfx>` (per-instance suffix; record key `account_id`; columns
account_id, score).

Ground truth by construction: a per-account point-in-time scan, cross-checked
against an independent vectorized `merge_asof` reference. Naive variants
(gates assert they differ): scoring on the latest revision overall
(current_values) and on the most recent revision within a sloppy T+7d window
(after_T_allowed).
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

KIND = "table"  # deliverable kind: table | dataset | answers | platform
SUMMARY = "Measures whether an agent can batch-score a whole entity population using point-in-time-correct feature values, an as-of join that must not look past the cutoff."

ORIGIN = pd.Timestamp("2026-02-01", tz="UTC")
N_ACCOUNTS = 80
N_DAYS = 60
FEATURES = ["f1", "f2", "f3"]
TABLE_BASE = "scores"  # per-instance: f"{TABLE_BASE}{instance_suffix(seed)}"

SPEC = {
    "columns": ["account_id", "score"],
    "ts_cols": [],
    "int_cols": [],
    "float_cols": ["score"],
    "sort_cols": ["account_id"],
}
VARIANT_DIAGNOSIS = {
    "current_values": "scored on each account's LATEST feature revision instead of "
    "the revision valid at T (most recent at or before T)",
    "after_T_allowed": "scored on revisions up to 7 days AFTER T — the as-of join "
    "must not look past T",
}


class GateError(RuntimeError):
    pass


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _score(hist: pd.DataFrame, model: dict) -> pd.Series:
    w = np.array([model["weights"][f] for f in FEATURES])
    z = hist[FEATURES].to_numpy() @ w + model["bias"]
    return pd.Series(np.round(_sigmoid(z), 6), index=hist.index)


def _scan_asof(hist: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Per-account scan: the most recent revision at or before `cutoff`."""
    rows = []
    for acct, grp in hist.groupby("account_id"):
        ok = grp[grp["event_time"] <= cutoff]
        if ok.empty:
            raise GateError(f"account {acct} has no revision at or before the cutoff")
        rows.append(ok.loc[ok["event_time"].idxmax()])
    return pd.DataFrame(rows).reset_index(drop=True)


def _merge_asof_ref(hist: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Independent vectorized reference: merge_asof at the cutoff."""
    accounts = pd.DataFrame({"account_id": sorted(hist["account_id"].unique())})
    accounts["event_time"] = cutoff
    return pd.merge_asof(
        accounts.sort_values("event_time"),
        hist.sort_values("event_time"),
        on="event_time",
        by="account_id",
        direction="backward",
    )


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    table = TABLE_BASE + instance_suffix(seed)
    t_day = int(rng.integers(38, 43))  # AS-OF day T (~40)
    cutoff = ORIGIN + pd.Timedelta(days=t_day, hours=int(rng.integers(8, 18)))

    # Revision days per account: guaranteed coverage before T, plus seeded
    # revisions in (T, T+7] and (T+7, 60) so both naive variants diverge.
    rows = []
    for i in range(N_ACCOUNTS):
        acct = f"A{i:04d}"
        days = set(int(d) for d in rng.integers(0, N_DAYS, int(rng.integers(4, 9))))
        days.add(int(rng.integers(0, 30)))  # at or before T
        if i % 4 == 0:
            days.add(t_day + 1 + int(rng.integers(0, 6)))  # in (T, T+7]
        if i % 4 == 1:
            days.add(t_day + 8 + int(rng.integers(0, N_DAYS - t_day - 9)))  # after T+7
        for d in sorted(days):
            ts = ORIGIN + pd.Timedelta(days=d, minutes=int(rng.integers(0, 1440)))
            rows.append([acct, ts.floor("s"), *np.round(rng.normal(0, 2, len(FEATURES)), 4)])
    hist = pd.DataFrame(rows, columns=["account_id", "event_time", *FEATURES])
    hist = hist.drop_duplicates(subset=["account_id", "event_time"]).reset_index(drop=True)

    model = {
        "weights": {
            f: round(float(w), 4) for f, w in zip(FEATURES, rng.normal(0, 0.8, len(FEATURES)))
        },
        "bias": round(float(rng.normal(0, 0.5)), 4),
    }

    def scores_of(asof_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"account_id": asof_df["account_id"], "score": _score(asof_df, model)})

    truth = canonicalize(scores_of(_scan_asof(hist, cutoff)), SPEC)

    # --- gates ---------------------------------------------------------------
    ref = canonicalize(scores_of(_merge_asof_ref(hist, cutoff)), SPEC)
    if digest(ref) != digest(truth):
        raise GateError(f"merge_asof reference disagrees with scan (seed={seed})")
    latest = hist.loc[hist.groupby("account_id")["event_time"].idxmax()]
    variants = {
        "current_values": canonicalize(scores_of(latest), SPEC),
        "after_T_allowed": canonicalize(
            scores_of(_scan_asof(hist, cutoff + pd.Timedelta(days=7))), SPEC
        ),
    }
    for name, v in variants.items():
        if digest(v) == digest(truth):
            raise GateError(f"variant {name!r} matches truth (seed={seed})")

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    emit = hist.copy()
    emit["event_time"] = emit["event_time"].dt.as_unit("ns").astype("int64") // 10**6
    emit.to_csv(out / "data" / "feature_history.csv", index=False)
    (out / "data" / "model.json").write_text(json.dumps(model, indent=2))
    t_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    t_ms = cutoff.value // 10**6  # as-of T in epoch milliseconds (matches event_time)
    (out / "data" / "scoring_request.md").write_text(
        "# Batch scoring request\n\n"
        f"Score EVERY account AS OF **T = {t_ms}** (epoch milliseconds; {t_iso}).\n\n"
        "For each account, use the feature values that were VALID AT time T — "
        "the most recent revision in data/feature_history.csv with "
        "`event_time` (epoch milliseconds) at or before T. Revisions after T "
        "must not influence any score.\n\n"
        "The model (data/model.json) is a logistic scorer:\n"
        "    score = sigmoid(w_f1*f1 + w_f2*f2 + w_f3*f3 + bias)\n"
        "rounded to 6 decimal places.\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains a feature-history export "
        "(data/feature_history.csv: account_id, event_time [epoch milliseconds], "
        "f1, f2, f3 — multiple revisions per account), a model (data/model.json: "
        "per-feature weights and a bias), and a scoring request "
        "(data/scoring_request.md) naming the as-of timestamp T.\n"
        "Batch-score EVERY account using the feature values that were VALID AT "
        "time T (the most recent revision at or before T; later revisions must "
        "not be used), with score = sigmoid(w_f1*f1 + w_f2*f2 + w_f3*f3 + bias) "
        "rounded to 6 decimal places.\n"
        f"Produce a feature table named `{table}`, version 1, on the platform, "
        "with record key `account_id` and exactly these columns: account_id, "
        "score. One row per account.\n"
        "Make the table's features available for low-latency lookup as well "
        "(online/real-time access), where the platform distinguishes the two.\n"
    )
    meta = {
        "family": "batch",
        "seed": seed,
        "table_name": table,
        "table_version": 1,
        "as_of": t_iso,
        "as_of_ms": int(t_ms),
        "spec": SPEC,
        "row_count": len(truth),
        "digest": digest(truth),
        "record_ids": truth["account_id"].tolist(),
        "variant_digests": {k: digest(v) for k, v in variants.items()},
        "variant_diagnosis": VARIANT_DIAGNOSIS,
        "spot_rows": truth.head(3).to_dict(orient="records"),
    }
    (out / "solution" / "truth.json").write_text(json.dumps(meta, indent=2))
    (out / "instance.json").write_text(json.dumps({"family": "batch", "seed": seed}, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-batch-selftest/{seed}"))
            print(f"[batch] seed={seed} rows={meta['row_count']} as_of={meta['as_of']} gates=OK")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
