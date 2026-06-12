"""Vector-search task (FTI sub-category: inference/vector_search) — generator.

Usage:
    python -m evals.inference.vector_search.generate --seed 7 --out /tmp/vs-7
    python -m evals.inference.vector_search.generate --selftest

World: 300 catalog items with 16-dim embeddings (`data/items.csv` — item_id,
embedding as a JSON-array string, label), built from a few seeded cluster
centers plus noise, and 25 query vectors (`data/queries.csv` — query_id,
embedding, same format). The agent must load the items into the platform's
vector-capable store named `items<sfx>` (per-instance suffix), run the
platform's NATIVE vector similarity search to get each query's top-5 nearest
items by Euclidean (L2) distance, and write `submission/answers.json`:

    {"store": "<store/index name>", "neighbors": {"<query_id>": [5 item_ids, nearest first]}}

Ground truth by construction: exact brute-force L2 k=5 per query, ranked
ascending — cross-checked against an independently-written second
implementation (scipy.spatial.distance.cdist). Naive variants (gates assert
they differ): ranking by cosine distance (cosine_metric) and by highest dot
product (dot_product).

ANN-stability: platforms answer through approximate indexes (HNSW etc.), so
the exact ranking must be the only defensible answer. `_stable` rejects (and
the generator deterministically resamples, from the same rng) any query where
(a) two of its top-6 distances sit within a 2% relative margin of each other
(adjacent check on the sorted distances covers all pairs), or (b)
dist(rank5)/dist(rank6) > 0.98 — i.e. the k-boundary is too tight. A gate
asserts the margins hold on the final world, so an approximate backend has no
excuse to return a different top-5 ordering.

Platform realization (encoded in the graders/adapters, asymmetric by design):
  hopsworks  — embedding index ON the feature group (hsfs EmbeddingIndex /
               EmbeddingFeature; `hops fg create --embedding col:dim[:metric]`),
               queried via fg.find_neighbors / `hops fg knn`.
  databricks — Vector Search endpoint + Direct Vector Access index
               (3-part UC name), upsert + query-index; SDK and CLI.
  sagemaker  — NO native vector similarity search on the allowed interface
               (AWS's blessed path, OpenSearch, is off-interface): the
               platform's best native representation is the vectors stored in
               an (online) feature group; neighbors are computed interface-side.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from evals.common import instance_suffix

N_ITEMS = 300
N_QUERIES = 25
N_CENTERS = 6
DIM = 16
K = 5
MARGIN = 0.02  # adjacent top-(K+1) distances must differ by >2% (see docstring)
TABLE_BASE = "items"  # per-instance: f"{TABLE_BASE}{instance_suffix(seed)}"

VARIANT_DIAGNOSIS = {
    "cosine_metric": "neighbors ranked by COSINE distance instead of Euclidean "
                     "(L2) — magnitude information was discarded",
    "dot_product": "neighbors ranked by HIGHEST DOT PRODUCT instead of smallest "
                   "Euclidean (L2) distance",
}


class GateError(RuntimeError):
    pass


def _rank(items: np.ndarray, ids: list[str], q: np.ndarray, metric: str) -> list[str]:
    """Top-K item ids for one query under `metric` (truth and naive variants
    share the loop, parameterized; best-first)."""
    if metric == "l2":
        d = np.linalg.norm(items - q, axis=1)
    elif metric == "cosine":
        d = 1.0 - (items @ q) / (np.linalg.norm(items, axis=1) * np.linalg.norm(q))
    else:  # dot_product: highest dot product first
        d = -(items @ q)
    order = np.argsort(d, kind="stable")[:K]
    return [ids[i] for i in order]


def _stable(items: np.ndarray, q: np.ndarray) -> bool:
    """True when the query's top-(K+1) L2 distances are pairwise separated by
    the relative margin — adjacent check on the ascending sort covers every
    pair, including the rank-5/rank-6 k-boundary."""
    d = np.sort(np.linalg.norm(items - q, axis=1))[:K + 1]
    return all(d[i + 1] > 0 and d[i] <= (1.0 - MARGIN) * d[i + 1] for i in range(K))


def _cdist_ref(items: np.ndarray, ids: list[str], queries: np.ndarray) -> list[list[str]]:
    """Independently-written second implementation (scipy cdist on the full
    query×item matrix) for the cross-check gate."""
    from scipy.spatial.distance import cdist
    d = cdist(queries, items, metric="euclidean")
    return [[ids[i] for i in np.argsort(row, kind="stable")[:K]] for row in d]


def generate(seed: int, out: Path) -> dict:
    rng = np.random.default_rng(seed)
    table = TABLE_BASE + instance_suffix(seed)

    # --- world: cluster centers + noise, 6-dp grid ----------------------------
    centers = rng.normal(0.0, 3.0, (N_CENTERS, DIM))
    item_cluster = rng.integers(0, N_CENTERS, N_ITEMS)
    items = np.round(centers[item_cluster] + rng.normal(0, 0.6, (N_ITEMS, DIM)), 6)
    item_ids = [f"I{i:04d}" for i in range(N_ITEMS)]

    def draw_query() -> np.ndarray:
        c = centers[int(rng.integers(0, N_CENTERS))]
        return np.round(c + rng.normal(0, 0.6, DIM), 6)

    queries = []
    for _ in range(N_QUERIES):
        q = draw_query()
        for _attempt in range(1000):
            if _stable(items, q):
                break
            q = draw_query()  # deterministic resample from the same rng
        else:
            raise GateError(f"no ANN-stable query after 1000 resamples (seed={seed})")
        queries.append(q)
    queries = np.array(queries)
    query_ids = [f"Q{i:03d}" for i in range(N_QUERIES)]

    # --- truth + naive variants ------------------------------------------------
    neighbors = {qid: _rank(items, item_ids, q, "l2")
                 for qid, q in zip(query_ids, queries)}
    variants = {
        "cosine_metric": {qid: _rank(items, item_ids, q, "cosine")
                          for qid, q in zip(query_ids, queries)},
        "dot_product": {qid: _rank(items, item_ids, q, "dot_product")
                        for qid, q in zip(query_ids, queries)},
    }

    # --- gates -------------------------------------------------------------------
    if not all(_stable(items, q) for q in queries):
        raise GateError(f"ANN-stability margin violated on the final world (seed={seed})")
    if _cdist_ref(items, item_ids, queries) != [neighbors[qid] for qid in query_ids]:
        raise GateError(f"scipy cdist reference disagrees with truth ranking (seed={seed})")
    for name, v in variants.items():
        if v == neighbors:
            raise GateError(f"variant {name!r} matches truth (seed={seed})")

    # --- write instance ----------------------------------------------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "solution").mkdir()
    pd.DataFrame({
        "item_id": item_ids,
        "embedding": [json.dumps([float(x) for x in row]) for row in items],
        "label": [f"c{c}" for c in item_cluster],
    }).to_csv(out / "data" / "items.csv", index=False)
    pd.DataFrame({
        "query_id": query_ids,
        "embedding": [json.dumps([float(x) for x in row]) for row in queries],
    }).to_csv(out / "data" / "queries.csv", index=False)
    (out / "data" / "schema.md").write_text(
        "# Schema\n\n"
        f"- **items.csv**: item_id (unique key), embedding (a JSON array of "
        f"{DIM} floats, as a string), label (item category)\n"
        f"- **queries.csv**: query_id (unique key), embedding (same format — "
        f"a JSON array of {DIM} floats)\n"
    )
    (out / "prompt.txt").write_text(
        "The directory data/ contains a catalog of items with vector "
        "embeddings (data/items.csv) and query vectors (data/queries.csv); "
        "see data/schema.md.\n"
        "Load the item embeddings into the platform's vector-capable store "
        f"named `{table}` (where the platform needs auxiliary resources — "
        "e.g. a vector search endpoint or index — name them with the same "
        f"`{table}` prefix).\n"
        "Use the platform's vector similarity search (its native ANN/KNN "
        "search where it has one) to retrieve, for EVERY query, the top-5 "
        "nearest items by Euclidean (L2) distance, nearest first.\n"
        "Write submission/answers.json as:\n"
        '    {"store": "<name of the store/index you created>", '
        '"neighbors": {"<query_id>": [<five item_ids, nearest first>]}}\n'
    )
    truth = {
        "family": "vector_search", "seed": seed,
        "table_name": table, "k": K, "metric": "l2", "dim": DIM,
        "n_items": N_ITEMS, "n_queries": N_QUERIES,
        "neighbors": neighbors,
        "variant_neighbors": variants,
        "variant_diagnosis": VARIANT_DIAGNOSIS,
    }
    (out / "solution" / "truth.json").write_text(json.dumps(truth, indent=2))
    (out / "instance.json").write_text(json.dumps(
        {"family": "vector_search", "seed": seed}, indent=2))

    # --- gates: the grade function accepts truth, rejects corruptions ----------
    from evals.inference.vector_search.grade import grade

    def run(answers: dict) -> dict:
        with tempfile.TemporaryDirectory(prefix="banter-vs-gate-") as td:
            run_dir = Path(td)
            (run_dir / "submission").mkdir()
            (run_dir / "submission" / "answers.json").write_text(json.dumps(answers))
            return grade(out, "none", run_dir)

    if not run({"store": table, "neighbors": neighbors})["success"]:
        raise GateError(f"reference answers fail the grade function (seed={seed})")
    spoiled = run({"store": table, "neighbors": variants["dot_product"]})
    if spoiled["success"]:
        raise GateError(f"dot_product variant passes the grade function (seed={seed})")
    if spoiled.get("diagnostic") != VARIANT_DIAGNOSIS["dot_product"]:
        raise GateError(f"dot_product variant not diagnosed (seed={seed})")
    partial = {qid: ids for qid, ids in list(neighbors.items())[:-1]}
    if run({"store": table, "neighbors": partial})["success"]:
        raise GateError(f"incomplete query coverage passes the grade function (seed={seed})")
    return truth


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        for seed in (1, 2, 3):
            meta = generate(seed, Path(f"/tmp/banter-vs-selftest/{seed}"))
            print(f"[vector_search] seed={seed} queries={meta['n_queries']} "
                  f"k={meta['k']} gates=OK")
        return 0
    if not args.out:
        ap.error("--out is required (or use --selftest)")
    print(json.dumps(generate(args.seed, args.out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
