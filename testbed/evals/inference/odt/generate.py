"""On-demand transformation task (FTI sub-category: inference/odt) — generator.

Usage:
    python -m evals.inference.odt.generate --seed 7 --out /tmp/odt-7
    python -m evals.inference.odt.generate --selftest

World: scoring requests (`data/requests.csv` — request-time parameters
including the request's coordinates) and stored account profiles
(`data/profiles.csv` — home coordinates + a precomputed base score). The
distance is an ON-DEMAND feature: it can only be computed at request time,
from the request's own parameters joined with the profile:

    distance_deg = round(sqrt((request_lat-home_lat)^2 + (request_lon-home_lon)^2), 6)
    score        = round(base_score - 0.1 * distance_deg, 6)

Deliverable: feature table `scored<sfx>` (per-instance suffix; record key
`request_id`; columns request_id, account_id, distance_deg, score).

Ground truth by construction: a per-row scan using math.sqrt, cross-checked
against an independently-written vectorized numpy reference. Naive variants
(gates assert they differ): adding the distance penalty instead of
subtracting it (swapped_sign) and using Manhattan |dlat|+|dlon| distance
(manhattan).

Rounding-tie avoidance: base_score and distance_deg both carry exactly 6
decimals, so the raw score base - 0.1*dist has a 7th decimal equal to dist's
6th-decimal digit. A 5 there puts the score EXACTLY on a 6-dp rounding
boundary, where equally-correct implementations legitimately disagree
(half-even on binary floats flips on invisible ulps — observed live 2026-06-11:
9/400 rows, a correct agent solution off truth by exactly 1e-6). `_detie`
nudges request coordinates until no distance ends in that digit; a gate
asserts the property holds on the final truth.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evals.common import canonicalize, digest, instance_suffix

KIND = "table"  # deliverable kind: table | dataset | answers | platform
SUMMARY = "Measures whether an agent can compute an on-demand feature at request time (a distance derived from request parameters joined with a stored profile) and apply it with the exact formula."

N_REQUESTS = 400
N_ACCOUNTS = 60
TABLE_BASE = "scored"  # per-instance: f"{TABLE_BASE}{instance_suffix(seed)}"
ORIGIN = pd.Timestamp("2026-04-01", tz="UTC")

SPEC = {
    "columns": ["request_id", "account_id", "distance_deg", "score"],
    "ts_cols": [],
    "int_cols": [],
    "float_cols": ["distance_deg", "score"],
    "sort_cols": ["request_id"],
}
VARIANT_DIAGNOSIS = {
    "swapped_sign": "the distance penalty was ADDED to base_score (it must be "
    "subtracted: score = base_score - 0.1*distance_deg)",
    "manhattan": "Manhattan |dlat|+|dlon| distance used instead of the Euclidean "
    "sqrt(dlat^2 + dlon^2)",
}


class GateError(RuntimeError):
    pass


def _scan(
    requests: pd.DataFrame, profiles: pd.DataFrame, metric: str = "euclidean", sign: float = -1.0
) -> pd.DataFrame:
    """Per-row scan (truth and naive variants share the loop, parameterized)."""
    prof = profiles.set_index("account_id")
    rows = []
    for _, r in requests.iterrows():
        p = prof.loc[r["account_id"]]
        dlat = r["request_lat"] - p["home_lat"]
        dlon = r["request_lon"] - p["home_lon"]
        if metric == "euclidean":
            dist = round(math.sqrt(dlat * dlat + dlon * dlon), 6)
        else:
            dist = round(abs(dlat) + abs(dlon), 6)
        score = round(p["base_score"] + sign * 0.1 * dist, 6)
        rows.append([r["request_id"], r["account_id"], dist, score])
    return pd.DataFrame(rows, columns=SPEC["columns"])


def _tie_digit(dist: float) -> bool:
    """True when a 6-dp distance ends in 5 — the digit that lands the raw
    score exactly on a 6-dp rounding boundary (see module docstring)."""
    return int(round(dist * 1e6)) % 10 == 5


def _detie(requests: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Nudge request latitudes (staying on the 6-dp grid) until no request's
    rounded distance ends in the tie digit."""
    prof = profiles.set_index("account_id")
    lat = requests["request_lat"].to_numpy().copy()
    lon = requests["request_lon"].to_numpy()
    for i, acct in enumerate(requests["account_id"]):
        p = prof.loc[acct]
        while True:
            dlat = lat[i] - p["home_lat"]
            dlon = lon[i] - p["home_lon"]
            # Same expression as _scan: an ulp difference in the distance
            # formula could flip the rounded digit and let a tie survive.
            if not _tie_digit(round(math.sqrt(dlat * dlat + dlon * dlon), 6)):
                break
            lat[i] = round(lat[i] + 2e-6, 6)
    requests = requests.copy()
    requests["request_lat"] = lat
    return requests


def _vectorized_ref(requests: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Independently-written vectorized reference (numpy hypot on the join)."""
    m = requests.merge(profiles, on="account_id", how="left")
    dist = np.round(np.hypot(m["request_lat"] - m["home_lat"], m["request_lon"] - m["home_lon"]), 6)
    score = np.round(m["base_score"] - 0.1 * dist, 6)
    return pd.DataFrame(
        {
            "request_id": m["request_id"],
            "account_id": m["account_id"],
            "distance_deg": dist,
            "score": score,
        }
    )


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    table = TABLE_BASE + instance_suffix(seed)
    profiles = pd.DataFrame(
        {
            "account_id": [f"A{i:04d}" for i in range(N_ACCOUNTS)],
            "home_lat": np.round(rng.uniform(-60, 60, N_ACCOUNTS), 6),
            "home_lon": np.round(rng.uniform(-170, 170, N_ACCOUNTS), 6),
            "base_score": np.round(rng.uniform(0.3, 0.95, N_ACCOUNTS), 6),
        }
    )
    acct_idx = rng.integers(0, N_ACCOUNTS, N_REQUESTS)
    requests = pd.DataFrame(
        {
            "request_id": [f"Q{i:05d}" for i in range(N_REQUESTS)],
            "account_id": profiles["account_id"].to_numpy()[acct_idx],
            "request_lat": np.round(
                profiles["home_lat"].to_numpy()[acct_idx] + rng.normal(0, 1.5, N_REQUESTS), 6
            ),
            "request_lon": np.round(
                profiles["home_lon"].to_numpy()[acct_idx] + rng.normal(0, 1.5, N_REQUESTS), 6
            ),
            "requested_at": [
                (ORIGIN + pd.Timedelta(minutes=int(m))).strftime("%Y-%m-%dT%H:%M:%SZ")
                for m in np.sort(rng.integers(0, 60 * 24 * 14, N_REQUESTS))
            ],
        }
    )

    requests = _detie(requests, profiles)
    truth = canonicalize(_scan(requests, profiles), SPEC)

    # --- gates ---------------------------------------------------------------
    if any(_tie_digit(d) for d in truth["distance_deg"]):
        raise GateError(f"rounding tie survived _detie (seed={seed})")
    ref = canonicalize(_vectorized_ref(requests, profiles), SPEC)
    if digest(ref) != digest(truth):
        raise GateError(f"vectorized reference disagrees with scan (seed={seed})")
    variants = {
        "swapped_sign": canonicalize(_scan(requests, profiles, sign=+1.0), SPEC),
        "manhattan": canonicalize(_scan(requests, profiles, metric="manhattan"), SPEC),
    }
    for name, v in variants.items():
        if digest(v) == digest(truth):
            raise GateError(f"variant {name!r} matches truth (seed={seed})")

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    requests.to_csv(out / "data" / "requests.csv", index=False)
    profiles.to_csv(out / "data" / "profiles.csv", index=False)
    (out / "data" / "schema.md").write_text(
        "# Schema\n\n"
        "- **requests.csv**: request_id (unique key), account_id, request_lat, "
        "request_lon (the request's coordinates — only known at request time), "
        "requested_at (ISO UTC)\n"
        "- **profiles.csv**: account_id (unique key), home_lat, home_lon, "
        "base_score (precomputed profile score)\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains scoring requests (data/requests.csv) and "
        "stored account profiles (data/profiles.csv); see data/schema.md.\n"
        "Each request carries request-time parameters (its coordinates) that "
        "must be combined with the stored profile via an ON-DEMAND "
        "transformation — a feature computed at request time, not precomputed:\n"
        "  - distance_deg = sqrt((request_lat - home_lat)^2 + "
        "(request_lon - home_lon)^2), rounded to 6 decimal places;\n"
        "  - score = base_score - 0.1 * distance_deg (using the rounded "
        "distance), rounded to 6 decimal places.\n"
        f"Produce a feature table named `{table}`, version 1, on the platform, "
        "with record key `request_id` and exactly these columns: request_id, "
        "account_id, distance_deg, score. One row per request.\n"
        "Make the table's features available for low-latency lookup as well "
        "(online/real-time access), where the platform distinguishes the two.\n"
    )
    meta = {
        "family": "odt",
        "seed": seed,
        "table_name": table,
        "table_version": 1,
        "spec": SPEC,
        "row_count": len(truth),
        "digest": digest(truth),
        "record_ids": truth["request_id"].tolist(),
        "variant_digests": {k: digest(v) for k, v in variants.items()},
        "variant_diagnosis": VARIANT_DIAGNOSIS,
        "spot_rows": truth.head(3).to_dict(orient="records"),
    }
    (out / "solution" / "truth.json").write_text(json.dumps(meta, indent=2))
    (out / "instance.json").write_text(json.dumps({"family": "odt", "seed": seed}, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-odt-selftest/{seed}"))
            print(f"[odt] seed={seed} rows={meta['row_count']} gates=OK")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
