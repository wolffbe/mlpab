# Databricks notebook source
import requests
import json

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
base_url = f"https://{host}"

vol_path = "/Volumes/workspace/mlpab6c8eeb/mlpab6c8eeb_data"
results = {}

# Try feature spec creation endpoints
feature_spec_payload = {
    "name": "workspace.mlpab6c8eeb.predictions7b586d_spec",
    "features": [
        {
            "table_name": "workspace.mlpab6c8eeb.predictions7b586d",
            "lookup_key": ["row_id"],
            "feature_columns": ["score"]
        }
    ]
}

for ep in [
    "/api/2.1/unity-catalog/feature-specs",
    "/api/2.0/unity-catalog/feature-specs",
    "/api/2.1/feature-specs",
    "/api/2.0/feature-specs",
    "/api/2.0/mlflow/databricks/feature-store/feature-specs",
]:
    try:
        r = requests.post(f"{base_url}{ep}", headers=headers, json=feature_spec_payload)
        results[f"POST {ep}"] = {"status": r.status_code, "body": r.text[:400]}
    except Exception as e:
        results[f"POST {ep}"] = {"error": str(e)}

# Also try serving endpoints with the right format
serving_payload = {
    "name": "mlpab6c8eeb_predictions7b586d",
    "config": {
        "served_entities": [
            {
                "entity_name": "workspace.mlpab6c8eeb.predictions7b586d_spec",
                "entity_version": "1",
                "scale_to_zero_enabled": True,
                "workload_size": "Small"
            }
        ]
    }
}
try:
    r = requests.post(f"{base_url}/api/2.0/serving-endpoints", headers=headers, json=serving_payload)
    results["POST serving with entity_name"] = {"status": r.status_code, "body": r.text[:400]}
except Exception as e:
    results["POST serving with entity_name"] = {"error": str(e)}

with open(f"{vol_path}/api_discovery2.json", "w") as f:
    json.dump(results, f, indent=2)
print("Done")
for k, v in results.items():
    print(f"{k}: status={v.get('status','?')} body={v.get('body','')[:100]}")
