# Databricks notebook source
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

st("create_model", lambda: w.registered_models.create(
    catalog_name=CAT, schema_name=SCH, name=NAME, comment="ftjob698f06"))
mv = ac.do("POST", "/api/2.0/mlflow/unity-catalog/model-versions/create",
           body={"name": FULL, "source": STAGE})["model_version"]
version = int(mv["version"]); loc = mv["storage_location"]
R["version"] = version; R["loc"] = loc

# Attempt 1: dbutils.fs.cp via control-plane path into the version storage.
def cp():
    dbutils.fs.cp(f"{STAGE}/finetuned_model.npz", f"{loc}/finetuned_model.npz")
    return dbutils.fs.ls(loc)
listing = st("dbutils_cp", cp)
if listing is not None:
    R["listing_after_cp"] = [f.path for f in listing]

# Attempt to finalize (works if content present, or if server copies from source).
fin = st("finalize", lambda: ac.do(
    "POST", "/api/2.0/mlflow/unity-catalog/model-versions/finalize",
    body={"name": FULL, "version": version}))
if fin: R["finalized_status"] = fin.get("model_version", {}).get("status")

# Verify final state.
def verify():
    m = w.model_versions.get(FULL, version)
    return {"status": str(m.status)}
R["verify"] = st("verify", verify)
try:
    R["files_in_loc"] = [f.path for f in dbutils.fs.ls(loc)]
except Exception as e:
    R["files_in_loc_err"] = repr(e)[:150]

dbutils.notebook.exit(json.dumps(R))
