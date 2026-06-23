---
name: hops-feature-views-training
description: Building feature views and point-in-time-correct training datasets in Hopsworks via CLI + SDK. Auto-invoke for feature views, training data, train/test splits, label selection, joins across feature groups, or model-dependent transformations.
---

# Feature views & training datasets

A **feature view** is a query selecting the exact features a model needs from one or more feature groups; it decouples training from the pipelines. **Model-dependent transformations** (scaling, encoding) attach HERE, not in the feature group. A **training dataset** is a point-in-time-correct snapshot computed from a feature view.

## Create a feature view (CLI)

```bash
hops fv create <name> --feature-group <fg>:1 \
  --join "<other_fg> LEFT <on>" \      # --join is repeatable
  --transform <fn>:<col> \             # --transform is repeatable (MDTs)
  --labels <label_col>
hops fv list
hops fv info <name> --version 1
```

## Compute a training dataset (CLI)

```bash
hops td compute <fv> <fv_version> --split "train:0.8,test:0.2"   # positional = FEATURE-VIEW version
hops td list <fv>                                                # TD version auto-increments — read it back
```

`hops td compute` takes the **feature-view** version as a required positional. The training-dataset version it writes auto-increments — never hardcode `1`; read it from `hops td list <fv>`.

## Point-in-time correctness (don't leak)

The feature store joins each label row to feature values **as of that row's event_time** when the FG has `event_time` set — no future values, no stale ones. To get this: set `event_time` on the source FGs, and let the feature view / `td compute` do the join. Do not hand-join on keys alone; that leaks future data.

## Read for training (SDK)

```python
fv = fs.get_feature_view(name="...", version=1)
X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)
# or read an existing TD version:
X, y = fv.get_training_data(training_dataset_version=<v>)
```
