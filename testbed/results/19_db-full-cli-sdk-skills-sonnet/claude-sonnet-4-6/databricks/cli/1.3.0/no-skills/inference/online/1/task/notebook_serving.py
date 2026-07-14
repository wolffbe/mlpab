# Databricks notebook source
import json, time, requests

schema = "workspace.mlpabcef85c"
table_name = "profilesaa70e4"
full_table_name = f"{schema}.{table_name}"
prefix = "mlpabcef85c"
endpoint_name = f"{prefix}-profile-ep"
feature_spec_name = f"{schema}.{prefix}_feat_spec"

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

results = {}

# COMMAND ----------
# Check what serving_endpoints can do
print("ServingEndpoints methods:", [m for m in dir(w.serving_endpoints) if not m.startswith('_')])
import databricks.sdk.service.serving as serving_module
serving_classes = sorted([x for x in dir(serving_module) if not x.startswith('_')])
print("Serving classes:", serving_classes)
results["serving_classes"] = serving_classes

# COMMAND ----------
# Check for feature spec / feature lookup related classes
feat_classes = [x for x in dir(serving_module) if 'feat' in x.lower() or 'Feature' in x or 'Lookup' in x]
print("Feature classes:", feat_classes)
results["feature_classes"] = feat_classes

# COMMAND ----------
# Try creating a feature spec via REST
print("=== Trying feature store / feature spec APIs ===")
for path in [
    "/api/2.0/feature-store/feature-specs",
    "/api/2.0/feature-engineering/feature-specs",
    "/api/2.1/feature-store/feature-specs",
    "/api/2.0/online-feature-lookup",
]:
    r = requests.get(f"{host}{path}", headers=headers)
    print(f"GET {path}: {r.status_code} {r.text[:200]}")
    results[f"rest_{path}"] = r.status_code

# COMMAND ----------
# Try creating a Feature Serving Endpoint with inline feature lookup
print("=== Creating Feature Serving Endpoint ===")

# Check if endpoint exists
ep_check = requests.get(f"{host}/api/2.0/serving-endpoints/{endpoint_name}", headers=headers)
print(f"Endpoint exists: {ep_check.status_code}")

if ep_check.status_code == 200:
    # Delete and recreate
    del_resp = requests.delete(f"{host}/api/2.0/serving-endpoints/{endpoint_name}", headers=headers)
    print(f"Delete: {del_resp.status_code}")
    time.sleep(5)

# Try creating with feature lookup entity
ep_payload = {
    "name": endpoint_name,
    "config": {
        "served_entities": [
            {
                "name": "feature-lookup",
                "entity_version": "1",
                "feature_spec_name": feature_spec_name
            }
        ]
    }
}

ep_resp = requests.post(f"{host}/api/2.0/serving-endpoints", headers=headers, json=ep_payload)
print(f"Endpoint create: {ep_resp.status_code} {ep_resp.text[:1000]}")
results["ep_create"] = {"status": ep_resp.status_code, "text": ep_resp.text[:500]}

# COMMAND ----------
# Try creating endpoint without feature spec (with served model)
# Maybe we need to create a model that returns features
ep2_name = f"{prefix}-model-ep"
ep2_check = requests.get(f"{host}/api/2.0/serving-endpoints/{ep2_name}", headers=headers)

if ep2_check.status_code != 200:
    # Create a simple endpoint with the databricks feature serving approach
    ep2_payload = {
        "name": ep2_name,
        "config": {
            "served_entities": [
                {
                    "name": "feature-table-lookup",
                    "feature_spec_name": feature_spec_name
                }
            ]
        }
    }
    ep2_resp = requests.post(f"{host}/api/2.0/serving-endpoints", headers=headers, json=ep2_payload)
    print(f"EP2 create: {ep2_resp.status_code} {ep2_resp.text[:500]}")

# COMMAND ----------
# Check the full endpoint list
eps = requests.get(f"{host}/api/2.0/serving-endpoints", headers=headers)
print(f"All endpoints: {eps.status_code}")
ep_data = eps.json()
print(json.dumps(ep_data, indent=2)[:2000])
results["all_endpoints"] = ep_data

# COMMAND ----------
# Try using Databricks Feature Engineering Client (newer version)
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()
    print(f"Feature Engineering Client created: {dir(fe)}")
    results["fe_client_available"] = True
except Exception as e:
    print(f"FE Client not available: {e}")
    results["fe_client_available"] = False

# COMMAND ----------
# Check if mlflow has feature store capabilities
try:
    import mlflow
    print(f"MLflow version: {mlflow.__version__}")
    print(f"MLflow modules: {[m for m in dir(mlflow) if 'feature' in m.lower() or 'online' in m.lower()]}")
    results["mlflow_version"] = mlflow.__version__
except Exception as e:
    results["mlflow_error"] = str(e)

# COMMAND ----------
print(json.dumps(results, indent=2)[:3000])
dbutils.notebook.exit(json.dumps(results)[:3000])
