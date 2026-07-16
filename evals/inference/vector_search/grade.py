"""Vector-search task — grader (platform kind).

Asserts:
    A1_format         — submission/answers.json parses, its `neighbors` map
        covers ALL query_ids, each entry is exactly 5 valid item_ids, no dupes;
    A2_neighbors      — every per-query ranked list matches the exact L2 truth
        (divergent rankings matching a precomputed naive variant — cosine
        distance / dot product — are named in `diagnostic`);
    A3_platform_state — the platform holds the vector store (state read via
        the adapter's `get_vector_store`; skipped-pass on adapter `none`).

Per-platform capability mapping for A3 (asymmetric by design):
    hopsworks  — feature group `items<sfx>` (any version) exists AND carries
                 an embedding index (fg.embedding_index is not None);
    databricks — a Vector Search index whose name contains `items<sfx>`
                 exists on some vector search endpoint (pure REST scan;
                 lenient on the catalog.schema prefix);
    sagemaker  — accepted in order: an Amazon S3 Vectors index `items<sfx>`
                 (the managed path — `s3vectors` is on the CLI allowlist,
                 native ANN via query-vectors), OR feature group `items<sfx>`
                 (vector storage, neighbors computed interface-side), OR an
                 InService endpoint `items<sfx>` (self-hosted FAISS-on-
                 endpoint, AWS's sagemaker-vector-store-microservice
                 pattern). SageMaker itself has no vector search; OpenSearch
                 stays off-interface.

Usage:
    python -m evals.inference.vector_search.grade --instance <dir> --adapter <hopsworks|databricks|sagemaker|none>
(cwd must be the run dir — the provider runs graders there.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.common import Suite, grade_platform_main, load_answers, state_checker, tally


def grade(instance_dir: Path, adapter: str, run_dir: Path) -> dict:
    truth = json.loads((Path(instance_dir) / "solution" / "truth.json").read_text())
    want = truth["neighbors"]
    valid_ids = {f"I{i:04d}" for i in range(truth["n_items"])}
    s = Suite()
    check = s.check
    diagnostic = None

    answers = load_answers(Path(run_dir) / "submission" / "answers.json")
    got = (answers or {}).get("neighbors") if isinstance(answers, dict) else None
    a1_ok, a1_detail = isinstance(got, dict), ""
    if not a1_ok:
        a1_detail = "answers.json must carry a 'neighbors' object {query_id: [5 item_ids]}"
    elif set(got) != set(want):
        a1_ok, a1_detail = (
            False,
            (
                f"neighbors must cover all {len(want)} query_ids "
                f"(missing: {sorted(set(want) - set(got))[:5]}, "
                f"unknown: {sorted(set(got) - set(want))[:5]})"
            ),
        )
    else:
        for qid, ids in got.items():
            ids = [str(i) for i in ids] if isinstance(ids, list) else []
            if len(ids) != truth["k"] or len(set(ids)) != truth["k"] or not set(ids) <= valid_ids:
                a1_ok, a1_detail = (
                    False,
                    (f"{qid}: expected exactly {truth['k']} distinct valid item_ids, got {ids!r}"),
                )
                break
    a1 = check("A1_format", a1_ok, a1_detail)

    if a1:
        norm = {qid: [str(i) for i in ids] for qid, ids in got.items()}
        bad = [qid for qid in want if norm[qid] != want[qid]]
        a2 = check(
            "A2_neighbors",
            not bad,
            f"{len(bad)}/{len(want)} queries diverge from the exact "
            f"L2 top-{truth['k']} ranking (e.g. {bad[:3]})"
            if bad
            else "",
        )
        if not a2:
            for vname, vmap in truth.get("variant_neighbors", {}).items():
                if norm == vmap:
                    diagnostic = truth.get("variant_diagnosis", {}).get(vname, vname)
                    break
    else:
        s.skip("A2_neighbors", "skipped: answers.json invalid (A1 failed)")

    if adapter == "none":
        s.skip("A3_platform_state", "no checker adapter (platform none) — skipped")
    else:
        st = state_checker(adapter).get_vector_store(truth["table_name"])
        check(
            "A3_platform_state",
            st.get("exists"),
            ""
            if st.get("exists")
            else f"vector store {truth['table_name']!r} not found on the platform: {st}",
        )

    return {
        "family": "vector_search",
        "seed": truth["seed"],
        **tally(s.asserts),
        "asserts": s.asserts,
        **({"diagnostic": diagnostic} if diagnostic else {}),
    }


def main(argv=None) -> int:
    return grade_platform_main("vector_search", grade, argv)


if __name__ == "__main__":
    sys.exit(main())
