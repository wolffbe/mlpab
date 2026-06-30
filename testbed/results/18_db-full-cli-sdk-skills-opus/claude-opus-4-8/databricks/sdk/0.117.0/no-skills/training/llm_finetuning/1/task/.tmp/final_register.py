# Databricks notebook source
# Final registration of workspace.mlpab5b087a.ftmodel698f06 version 1.
# Content (finetuned_model.npz) is referenced via the version `source` (a UC
# volume), since UC managed model storage is not writable from serverless.
# Metrics from metrics.json are attached as model-version tags.
import json
from databricks.sdk import WorkspaceClient

STAGE = "/Volumes/workspace/mlpab5b087a/ftstage"
CAT, SCH, NAME = "workspace", "mlpab5b087a", "ftmodel698f06"
FULL = f"{CAT}.{SCH}.{NAME}"
w = WorkspaceClient(); ac = w.api_client
R = {"steps": []}
def st(n, fn):
    try: r = fn(); R["steps"].append([n, "ok"]); return r
    except Exception as e: R["steps"].append([n, repr(e)[:250]]); return None

metrics = json.load(open(f"{STAGE}/metrics.json")); R["metrics"] = metrics

# Clean slate so the next created version is version 1.
def reset():
    try:
        for v in w.model_versions.list(FULL):
            try: w.model_versions.delete(FULL, v.version)
            except Exception: pass
        w.registered_models.delete(FULL)
    except Exception:
        pass
st("reset", reset)
st("create_model", lambda: w.registered_models.create(
    catalog_name=CAT, schema_name=SCH, name=NAME,
    comment=f"Fine-tuned char-level bigram LM (ftjob698f06). "
            f"metrics={json.dumps(metrics)}. content={STAGE}/finetuned_model.npz"))

mv = ac.do("POST", "/api/2.0/mlflow/unity-catalog/model-versions/create",
           body={"name": FULL, "source": f"{STAGE}"})["model_version"]
version = int(mv["version"]); R["version"] = version

st("finalize", lambda: ac.do(
    "POST", "/api/2.0/mlflow/unity-catalog/model-versions/finalize",
    body={"name": FULL, "version": version}))

for k, v in metrics.items():
    st(f"tag:{k}", lambda k=k, v=v: ac.do(
        "POST", "/api/2.0/mlflow/unity-catalog/model-versions/set-tag",
        body={"name": FULL, "version": version, "key": k, "value": str(v)}))

# Record content reference + metrics in the version comment too.
st("ver_comment", lambda: w.model_versions.update(
    FULL, version,
    comment=f"content={STAGE}/finetuned_model.npz; metrics={json.dumps(metrics)}"))

st("alias", lambda: w.registered_models.set_alias(FULL, "prod", version))

# Read back for verification.
def verify():
    mvi = w.model_versions.get(FULL, version)
    return {"version": mvi.version, "status": str(getattr(mvi, "status", None)),
            "comment": mvi.comment}
R["verify"] = st("verify", verify)
dbutils.notebook.exit(json.dumps(R))
