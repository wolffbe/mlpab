---
name: hops-inference
description: Batch and online (real-time) inference in Hopsworks — model registry, KServe deployments, feature-vector retrieval, predictor scripts. Auto-invoke for model deployment, online serving, batch scoring, get_feature_vector, predictor.py, or serving-skew issues.
---

# Inference pipelines

An inference pipeline reads features (offline for batch, online for real-time), applies a registered model, and writes/returns predictions. What you deploy for real-time is the *pipeline* (a KServe endpoint), not the model alone.

## Register & inspect (CLI)

```bash
hops model list
hops model info <name> --version 1
hops deployment status ; hops deployment delete <name>
```

## Batch inference

Read a training/batch dataset from the feature view in your own environment, call `model.predict`, write predictions back (often to a predictions feature group). No serving image involved, so no version skew.

```python
fv = fs.get_feature_view(name="...", version=1)
df = fv.get_batch_data()          # offline features
preds = model.predict(df)
```

## Online inference (real-time)

Needs an **online-enabled** feature view backed by online FGs. Build the feature vector at request time:

```python
fv.init_serving()                          # or fv.init_batch_scoring()
vec = fv.get_feature_vector({"id": key})   # precomputed online features
vecs = fv.get_feature_vectors([...])       # batched
```

Deploy via the SDK (`model.deploy(...)` → KServe). A `predictor.py` is required for custom frameworks/python deployments; sklearn/PyTorch/etc. can use the built-in server.

## Serving skew (a common, avoidable failure)

The KServe sklearn image pins a specific scikit-learn. A model pickled with a different version can fail to unpickle on the endpoint. Either pin training to the serving image's version, or deploy as a `python` deployment with a cloned env carrying your training versions + `predictor.py`. Batch/interactive inference run in your own env and have no skew.
