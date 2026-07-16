"""Recsys retrieval+ranking task (FTI sub-category: inference/recsys) — generator.

Usage:
    python -m evals.inference.recsys.generate --seed 7 --out /tmp/recsys-7
    python -m evals.inference.recsys.generate --selftest

World: seeded interactions (`data/interactions.csv` — user/item pairs the
user has already seen), item embeddings (`data/item_embeddings.csv`) and user
embeddings (`data/user_embeddings.csv`), both 8-dimensional (two-tower shape).
Deliverable: feature table `recs<sfx>` (per-instance suffix; record key `rec_id` formatted
"<user_id>#<rank>"; columns rec_id, user_id, rank 1..5, item_id) — the top-5
items per user by dot(user_emb, item_emb), EXCLUDING items the user has
already interacted with, ties broken by ascending item_id.

The world makes both rules bite: every user's single highest-dot item is
seeded into their interactions (so include_seen diverges), and several item
pairs share identical embedding vectors (exact score ties, so the tie-break
direction diverges). Ground truth by construction: a per-user scan,
cross-checked against an independent vectorized matmul reference. Naive
variants (gates assert they differ): include_seen and ties_desc.
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
SUMMARY = (
    "Measures whether an agent can run a two-tower retrieval and ranking step correctly: dot- "
    "product relevance, exclusion of already-seen items, and a deterministic tie-break."
)

N_USERS = 40
N_ITEMS = 60
DIM = 8
TOP_K = 5
N_DUP_PAIRS = 8  # item pairs with identical embeddings → exact ties
TABLE_BASE = "recs"  # per-instance: f"{TABLE_BASE}{instance_suffix(seed)}"

SPEC = {
    "columns": ["rec_id", "user_id", "rank", "item_id"],
    "ts_cols": [],
    "int_cols": ["rank"],
    "sort_cols": ["rec_id"],
}
VARIANT_DIAGNOSIS = {
    "include_seen": "already-interacted items were recommended (they must be excluded)",
    "ties_desc": "score ties broken by DESCENDING item_id (the rule is ascending)",
}


class GateError(RuntimeError):
    pass


def _scan_topk(
    users: pd.DataFrame,
    items: pd.DataFrame,
    seen: dict[str, set],
    exclude_seen: bool = True,
    ties_ascending: bool = True,
) -> pd.DataFrame:
    """Per-user scan (truth and naive variants share it, parameterized)."""
    ecols = [f"e{i}" for i in range(1, DIM + 1)]
    rows = []
    for _, u in users.iterrows():
        uvec = u[ecols].to_numpy(dtype=float)
        cands = []
        for _, it in items.iterrows():
            if exclude_seen and it["item_id"] in seen.get(u["user_id"], set()):
                continue
            score = float(np.dot(uvec, it[ecols].to_numpy(dtype=float)))
            cands.append((score, it["item_id"]))
        if ties_ascending:
            cands.sort(key=lambda c: (-c[0], c[1]))
        else:
            cands.sort(key=lambda c: c[1], reverse=True)  # desc id …
            cands.sort(key=lambda c: -c[0])  # … then stable by score
        for rank, (_, item_id) in enumerate(cands[:TOP_K], start=1):
            rows.append([f"{u['user_id']}#{rank}", u["user_id"], rank, item_id])
    return pd.DataFrame(rows, columns=SPEC["columns"])


def _matmul_ref(users: pd.DataFrame, items: pd.DataFrame, seen: dict[str, set]) -> pd.DataFrame:
    """Independent vectorized reference: U @ I.T, seen masked, lexsort ranks."""
    ecols = [f"e{i}" for i in range(1, DIM + 1)]
    U = users[ecols].to_numpy(dtype=float)
    I = items[ecols].to_numpy(dtype=float)
    item_ids = items["item_id"].to_numpy()
    scores = U @ I.T
    rows = []
    for ui, user_id in enumerate(users["user_id"]):
        s = scores[ui].copy()
        mask = np.isin(item_ids, list(seen.get(user_id, set())))
        s[mask] = -np.inf
        order = np.lexsort((item_ids, -s))  # score desc, id asc
        for rank, ii in enumerate(order[:TOP_K], start=1):
            rows.append([f"{user_id}#{rank}", user_id, rank, item_ids[ii]])
    return pd.DataFrame(rows, columns=SPEC["columns"])


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    table = TABLE_BASE + instance_suffix(seed)
    ecols = [f"e{i}" for i in range(1, DIM + 1)]
    items = pd.DataFrame(np.round(rng.normal(0, 1, (N_ITEMS, DIM)), 4), columns=ecols)
    items.insert(0, "item_id", [f"I{i:04d}" for i in range(N_ITEMS)])
    # identical-embedding item pairs → exact score ties for every user
    dup_idx = rng.choice(N_ITEMS, 2 * N_DUP_PAIRS, replace=False)
    for a, b in dup_idx.reshape(-1, 2):
        items.loc[b, ecols] = items.loc[a, ecols].to_numpy()
    users = pd.DataFrame(np.round(rng.normal(0, 1, (N_USERS, DIM)), 4), columns=ecols)
    users.insert(0, "user_id", [f"U{i:04d}" for i in range(N_USERS)])

    # seen sets: each user's single best item (forces the exclusion rule to
    # matter) + a handful of random items
    I = items[ecols].to_numpy(dtype=float)
    seen: dict[str, set] = {}
    inter_rows = []
    for ui, u in users.iterrows():
        s = u[ecols].to_numpy(dtype=float) @ I.T
        best = items["item_id"].iloc[int(np.argmax(s))]
        picks = {best} | set(
            items["item_id"].iloc[rng.integers(0, N_ITEMS, int(rng.integers(4, 10)))]
        )
        seen[u["user_id"]] = picks
        inter_rows += [[u["user_id"], it] for it in sorted(picks)]
    interactions = pd.DataFrame(inter_rows, columns=["user_id", "item_id"])

    truth = canonicalize(_scan_topk(users, items, seen), SPEC)

    # --- gates ---------------------------------------------------------------
    ref = canonicalize(_matmul_ref(users, items, seen), SPEC)
    if digest(ref) != digest(truth):
        raise GateError(f"matmul reference disagrees with scan (seed={seed})")
    variants = {
        "include_seen": canonicalize(_scan_topk(users, items, seen, exclude_seen=False), SPEC),
        "ties_desc": canonicalize(_scan_topk(users, items, seen, ties_ascending=False), SPEC),
    }
    for name, v in variants.items():
        if digest(v) == digest(truth):
            raise GateError(f"variant {name!r} matches truth (seed={seed})")

    # --- write instance --------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    interactions.to_csv(out / "data" / "interactions.csv", index=False)
    items.to_csv(out / "data" / "item_embeddings.csv", index=False)
    users.to_csv(out / "data" / "user_embeddings.csv", index=False)
    (out / "data" / "schema.md").write_text(
        "# Schema\n\n"
        "- **interactions.csv**: user_id, item_id — items the user has ALREADY "
        "interacted with\n"
        f"- **item_embeddings.csv**: item_id (unique key), e1..e{DIM}\n"
        f"- **user_embeddings.csv**: user_id (unique key), e1..e{DIM}\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains seeded user/item interactions "
        "(data/interactions.csv) and two-tower embeddings "
        "(data/user_embeddings.csv, data/item_embeddings.csv); see "
        "data/schema.md.\n"
        f"For EVERY user, retrieve and rank the top-{TOP_K} recommended items: "
        "relevance = the dot product of the user's embedding with the item's "
        "embedding; EXCLUDE items the user has already interacted with "
        "(interactions.csv); rank 1 is the highest-relevance item; break exact "
        "score ties by ASCENDING item_id.\n"
        f"Produce a feature table named `{table}`, version 1, on the platform, "
        'with record key `rec_id` formatted "<user_id>#<rank>" (e.g. '
        '"U0003#1") and exactly these columns: rec_id, user_id, rank '
        f"(integer 1..{TOP_K}), item_id. {TOP_K} rows per user.\n"
        "Make the table's features available for low-latency lookup as well "
        "(online/real-time access), where the platform distinguishes the two.\n"
    )
    meta = {
        "family": "recsys",
        "seed": seed,
        "table_name": table,
        "table_version": 1,
        "spec": SPEC,
        "row_count": len(truth),
        "digest": digest(truth),
        "record_ids": truth["rec_id"].tolist(),
        "variant_digests": {k: digest(v) for k, v in variants.items()},
        "variant_diagnosis": VARIANT_DIAGNOSIS,
        "spot_rows": truth.head(3).to_dict(orient="records"),
    }
    (out / "solution" / "truth.json").write_text(json.dumps(meta, indent=2))
    (out / "instance.json").write_text(json.dumps({"family": "recsys", "seed": seed}, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/mlpab-recsys-selftest/{seed}"))
            print(f"[recsys] seed={seed} rows={meta['row_count']} gates=OK")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
