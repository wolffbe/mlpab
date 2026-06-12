# Evals — generated FTI tasks for ML-platform agents

Generated evals measuring how well LLM agents operate ML platforms
(Hopsworks, Databricks, SageMaker) through their native interfaces (SDK, CLI)
across a **standard FTI pipeline** (Dowling, *Building Machine Learning
Systems with a Feature Store*, O'Reilly 2026). One task per FTI sub-category —
no difficulty tiers; each task is the honest version of its sub-category: the
requirement is stated, the world contains the realistic complication, and the
naive solution provably fails an assert.

## Research questions (slices of one results table)

- **RQ1 — Interfaces**: SDK vs CLI, per platform.
- **RQ2 — Models**: Claude family + GPT/GPT-mini/GPT-nano, SDK fixed, with
  the local baseline as ceiling.
- **RQ3 — Skills**: the vendor's official skill bundle vs none.

Configs: `configs/treatments/<platform>/<model>-{skills,no-skills}.yaml`
(+ `configs/treatments/local/<model>.yaml` — the no-platform baseline).
Different-platform configs are parallel-safe; same-platform configs and the
per-model multi-platform baselines run sequentially (per-run teardown sweeps
the platform). Every config covers all implemented tasks; `n` repeats give
fresh seeded instances per repeat.

## Design principles

1. **Ground truth by construction** — the generator seeds the world, so it
   knows the answer before the agent starts; truth is computed by committed
   pandas/numpy code, never by an LLM, never through the platform under test.
2. **Assertion-suite grading** — binary success: the frozen suite replayed
   against the platform's own read paths goes green. `asserts_passed/
   asserts_total` land in results.csv; the full report with per-assert
   results and a named failure diagnosis lands in the run's grading.json.
3. **Validity gates at generation time** — the committed reference solution
   must reproduce the truth digest and every precomputed naive variant must
   differ; instances failing either gate are rejected (this caught real
   design bugs during development).
4. **Platform-neutral prompts** — outcomes in domain language ("feature
   table", "recurring job", "model registry"), never platform primitives;
   mapping language → mechanism is part of what's measured.
5. **Fresh instances per run** — deterministic seeds per
   (config, category, task, attempt).
6. **Grader/run separation** — the answer key lives in a sibling
   `.<task>.private/` dir outside the agent's sandbox; grading runs outside
   the boundary through per-platform checker adapters.

## Deliverable kinds

| Kind | Deliverable | Graded via |
|---|---|---|
| `table` | a feature table on the platform | adapter table read (Hopsworks query service / Databricks SQL / SageMaker online `batch_get_record` on the truth's `row_id`s) |
| `dataset` | a versioned training dataset | adapter dataset read (feature view TD / `<name>_v<N>` table / S3 convention) |
| `answers` | `submission/answers.json` | local comparison against the seed |
| `platform` | mixed: tables/answers + platform STATE | adapter state reads: `get_model`, `get_job`, `get_endpoint`, `get_alert` |

Platform `none` (the local baseline) maps tables/datasets to
`submission/<name>.csv` and skips platform-state asserts (recorded as
skipped-pass); see the local-baseline section of `prompts/agent.md`.

## Task roster — all implemented (24)

### F — feature pipelines
| Task | Kind | The trap the asserts catch |
|---|---|---|
| `ingest` | table | re-delivered overlap kept twice; epoch-ms timestamps unparsed |
| `backfill` | table | load-order upsert loses out-of-order corrections |
| `mit` | table | 7-day window off-by-one (8 days / self-exclusive) |
| `validate` | table | violations ingested; rejected row_ids unreported |
| `incremental_load` | platform | missing increment; no recurring job registered |
| `full_reload` | table | stale rows survive the schema-breaking re-create |
| `training_data` | dataset | non-PIT join / ignored late export / future leak (each diagnosed) |
| `leakage` | answers | wrong leaky feature |

### T — training pipelines
| Task | Kind | Trap |
|---|---|---|
| `train` | platform | provided deterministic script not run as a platform job; undertrained variant |
| `mdt` | table | scaler stats fitted on all data, or per split (both diagnosed) |
| `register` | platform | wrong/missing metrics; no registry entry |
| `llm_finetuning` | platform | skipped fine-tune (base loss reported as eval loss) / undertrained adapter (N/10 iterations) — each diagnosed |

`llm_finetuning` platform realization: the provided deterministic LoRA-style
script (rank-4 adapter on a frozen bigram-LM checkpoint) runs as a platform
job `ftjob<sfx>` and the result lands in the registry as `ftmodel<sfx>` v1 —
graded via the existing `get_job` + `get_model` state reads; no new adapter
reads needed.

### I — inference pipelines
| Task | Kind | Trap |
|---|---|---|
| `batch` | table | scored on latest revision instead of valid-at-T |
| `online` | platform | vectors not retrieved through the online path |
| `odt` | table | request-time feature swapped sign / wrong distance |
| `skew` | answers | wrong diverging feature |
| `llm_serving` | platform | scorer responses wrong; endpoint absent |
| `recsys` | table | seen items included; ties broken descending |
| `vector_search` | platform | cosine/dot-product ranking instead of exact L2; no vector store on the platform |

`vector_search` platform realization (asymmetric by design; truth is
ANN-stable — a 2% relative margin between adjacent top-6 distances is
enforced at generation, so approximate indexes have no excuse to return a
different exact top-5):
- **hopsworks** — native: the embedding index lives ON the feature group
  (hsfs `EmbeddingIndex`/`EmbeddingFeature`, queried via `fg.find_neighbors`;
  CLI: `hops fg create --embedding col:dim[:metric]` + `hops fg knn`).
- **databricks** — native: a Vector Search endpoint + a Direct Vector Access
  index (3-part UC name) accepting upserted vectors, queried with
  query-index; SDK and CLI.
- **sagemaker** — SageMaker itself has no vector similarity search; the
  managed path on this interface is **Amazon S3 Vectors** (`aws s3vectors`:
  vector buckets + indexes, put-vectors/query-vectors; on the CLI allowlist,
  needs the `s3vectors:*` grant from `banter-policy.json`; verified live in
  eu-north-1). The grader accepts, in order: an S3 Vectors index named
  `items<sfx>` (native ANN), the vectors stored in an (online) feature group
  with neighbors computed interface-side, or an InService endpoint
  `items<sfx>` (self-hosted FAISS-on-endpoint, per AWS's own
  sagemaker-vector-store-microservice sample). OpenSearch stays
  off-interface. SDK-arm caveat: the `sagemaker` package has no vector API,
  so the managed path from the SDK means direct boto3 `s3vectors` calls —
  those don't count as interface calls in the metrics.

### Ops — observability & operations
| Task | Kind | Trap |
|---|---|---|
| `drift` | answers | wrong feature/onset |
| `prediction_monitoring` | answers | wrong prediction-drift onset |
| `scheduled_jobs` | platform | job missing or not scheduled |
| `alerting` | platform | failing job has no alert configured |
| `lineage` | platform | outer-join fill; wrong derivation answer |

Categories in configs: `feature` / `training` / `inference` / `ops` — note
`mit` (model-independent) is feature-stage, `mdt` (model-dependent,
train-split-only statistics) is training-stage, per the book's distinction.

## Layout

`evals/{feature,training,inference,ops}/<family>/` — one package per task
family, grouped by FTI category; shared plumbing in `evals/common.py` and
`evals/adapters/`. `docs/evals.md` (this file) is the canonical doc (the old
evals/README.md is gone).

## Adapters (`evals/adapters/`)

Per-platform checkers used ONLY by graders, with the grader's credentials,
outside the agent boundary. Data reads: feature tables, training datasets,
online records. State reads (best-effort dicts asserted on `exists` + detail
keys): `get_model` (Hopsworks registry / Databricks UC-then-MLflow /
SageMaker model package groups), `get_job` (Hopsworks jobs+executions /
Databricks Jobs API / SageMaker pipelines+training/processing jobs),
`get_endpoint` (deployments / serving endpoints / endpoints), `get_alert`
(Hopsworks alert routes / Databricks job notifications / CloudWatch alarms),
`get_vector_store` (Hopsworks feature-group embedding index / Databricks
Vector Search index scan over endpoints, pure REST / SageMaker: S3 Vectors
index, else feature-group existence, else InService endpoint).
**Hopsworks was live-validated end-to-end** (reference PIT solution graded
4/4 through the cluster); Databricks (SQL read path, warehouse discovery) and
SageMaker (auth, feature-group describe) are live-probed.

## Status & operational notes

- All 24 generators: selftests with gates green; round trips (independent
  solve from staged data → grade) pass and spoiled solves are rejected with
  named diagnoses.
- mlkit (the fake platform) and the smoketest configs are REMOVED; the local
  baseline configs replace them as the no-credentials sanity check.
- Skills are fetched at runtime from pinned vendor repos
  (configs/platforms/<p>/skills.yaml: repo+ref+path → build home).
- Removed from the testbed: MLE-bench (thesis motivating pilot only),
  autoresearch (future work), difficulty tiers, the researcher/engineer
  split, the maximize/minimize goal notion.
- Databricks setup.py does not yet pre-provision a SQL warehouse (the adapter
  falls back to the first existing one); codex auth pending for GPT columns.
