---
name: hops-feature-groups
description: Creating, inserting into, and managing Hopsworks feature groups (tables) via CLI + Python SDK. Auto-invoke for feature pipelines, feature engineering, ingestion, backfill, incremental loads, or questions about online/offline, types, event_time, and materialization.
---

# Feature groups — the fast, correct write

A feature pipeline applies **model-independent transformations** (aggregations, joins, parsing) and writes **untransformed, reusable** features. Do NOT bake model-dependent transforms (scaling, one-hot) into a feature group — those belong in the feature view. Storing them makes data non-reusable and forces rewrites.

## Decisions to settle up front (cheaper than fixing later)

- **Online or offline?** Default **offline** (`online_enabled=False`). Use **online + offline** (`online_enabled=True`) only when low-latency serving / `get_feature_vector` is needed.
- **Primary key + event_time.** Set `event_time` to when the value was *valid* (not ingest time) for any time-series feature — this is what enables point-in-time-correct training data. Omit only for immutable data.
- **Provenance.** If this FG derives from others, pass `parents=[fg1, fg2]` (the objects) at creation.

## Types (a wrong type loops forever — pick once)

- Scalars: `int`, `bigint`, `float`, `double`, `boolean`, `string`, `date`, `timestamp`, `binary`.
- Composite: `array<type>`, `struct<...>` — these write online too (Avro-encoded), no need to flatten.
- `decimal` is **not** supported → use `double`, or `string` for exact precision.
- Timestamps as epoch-ms `bigint` are accepted; let the schema be inferred from the DataFrame unless you must pin a type via `features=[Feature(...)]`.

## Minimal write (SDK)

```python
fs = project.get_feature_store()
fg = fs.get_or_create_feature_group(
    name="...", version=1, primary_key=["id"], event_time="ts",
    online_enabled=False,
)
fg.insert(df)            # registers on first insert; blocks for offline write
```

## Confirm with the CLI (don't re-insert blindly)

```bash
hops fg list
hops fg info <name> --version 1
hops fg preview <name> --version 1 --n 10
hops fg stats <name> --version 1
```

Re-running create+insert with identical data risks duplicate or conflict errors. Check `hops fg list` first; insert is an upsert on the primary key for online FGs (one row per key).
