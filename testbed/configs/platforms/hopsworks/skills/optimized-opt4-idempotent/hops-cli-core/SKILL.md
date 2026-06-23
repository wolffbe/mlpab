---
name: hops-cli-core
description: Core workflow for the Hopsworks `hops` CLI — login, project/feature-store handle, and the inspect-before-write loop. Auto-invoke at the start of ANY Hopsworks task and whenever you are unsure of current state (does a feature group/view/model exist, what is its schema, what version).
---

# Hopsworks via the `hops` CLI — fast path

`hops` is authenticated from `HOPSWORKS_API_KEY` + `HOPSWORKS_HOST` in the env. There is exactly ONE project; `hops` and `hopsworks.login()` select it with no prompt. Do not create projects.

**Prefer the CLI for everything it can do** (inspect, create FG/FV, compute training data, preview, stats). Drop to the Python SDK only for work the CLI cannot express: model training, KServe deployment, on-demand/model-dependent transformation code.

## The loop: inspect → act → confirm

Cheap reads beat guessing. Before writing Python and after every mutating step, inspect with the CLI — no Spark session needed, low token cost:

```bash
hops fg list                                # names, versions, STORE (own vs shared)
hops fg info <name> --version 1             # id, online flag, primary key, event_time
hops fg features <name> --version 1         # schema + key/partition flags
hops fg preview <name> --version 1 --n 10   # first rows (flag is --n)
hops fg stats <name> --version 1            # null counts / ranges — catch bad data early
hops fv list ; hops td list                 # feature views / training datasets
hops model list ; hops job list ; hops env list
```

Run `hops --help` and `hops <group> --help` once to confirm the exact flags rather than assuming them.

## Rules that avoid wasted retries

- A feature group registers server-side on its **first insert**, not at create. Until then `fg.id` is `None` and it is absent from `hops fg list`.
- Versions are explicit. Read a version back from `list` instead of hardcoding `1` — training-dataset and FV versions auto-increment.
- `hops fg list` STORE column: imported/public FGs live in a **shared** store; pass `--featurestore <store>` to read them.
- Don't repeat an identical failing call. Read the error, change the input (type, version, flag), then retry once.
- Reading back via the SDK uses `.read()` / `fg.show(n)`; `hops sql` / Trino may be unavailable to external clients — use `hops fg preview` to verify data landed.
