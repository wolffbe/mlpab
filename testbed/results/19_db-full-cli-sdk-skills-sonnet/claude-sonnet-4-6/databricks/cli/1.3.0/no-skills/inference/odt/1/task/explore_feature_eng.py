# Databricks notebook source
# COMMAND ----------
import json
import requests

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = {}

# Check databricks-feature-engineering package
try:
    import databricks.feature_engineering as fe
    results["fe_version"] = str(fe.__version__ if hasattr(fe, '__version__') else "no version")
    results["fe_dir"] = [x for x in dir(fe) if not x.startswith('_')]
except Exception as e:
    results["fe_import"] = str(e)

# Check if FeatureEngineeringClient is available
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    fec = FeatureEngineeringClient()
    fec_methods = [m for m in dir(fec) if not m.startswith('_') and callable(getattr(fec, m))]
    results["fec_methods"] = fec_methods
except Exception as e:
    results["fec_error"] = str(e)

# Try feature serving API paths
for path in [
    "/api/2.0/feature-serving",
    "/api/2.0/feature-store/feature-tables",
    "/api/2.0/online-store",
    "/api/2.0/unity-catalog/feature-tables",
    "/api/2.0/catalog/feature-tables",
]:
    resp = requests.get(f"https://{host}{path}", headers=headers)
    results[f"GET {path}"] = f"{resp.status_code}: {resp.text[:80]}"

dbutils.notebook.exit(json.dumps(results))
