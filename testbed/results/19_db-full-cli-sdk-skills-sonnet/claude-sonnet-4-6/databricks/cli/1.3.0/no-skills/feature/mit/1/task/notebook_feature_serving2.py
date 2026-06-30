# Databricks notebook source
# COMMAND ----------
import json, requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

# Try creating a Databricks Feature Serving Endpoint
# This is the Databricks mechanism for low-latency feature lookup from UC tables
payload = {
    "name": "mlpabf1452c_featuresb1ea93",
    "config": {
        "served_entities": [{
            "entity_name": "workspace.mlpabf1452c.featuresb1ea93",
            "entity_version": "1",
            "workload_size": "Small",
            "scale_to_zero_enabled": True
        }]
    }
}

try:
    r = requests.post(
        f"{host}/api/2.0/serving-endpoints",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=30
    )
    results['serving_endpoint'] = {"status": r.status_code, "body": r.text[:500]}
except Exception as e:
    results['serving_endpoint'] = {"error": str(e)}

# Also try a direct feature lookup endpoint
payload2 = {
    "name": "mlpabf1452c_featuresb1ea93_lookup",
    "spec": {
        "features": [{
            "table_name": "workspace.mlpabf1452c.featuresb1ea93",
            "lookup_key": ["row_id"],
            "timestamp_lookup_key": "event_time"
        }]
    }
}

try:
    r2 = requests.post(
        f"{host}/api/2.0/feature-serving/endpoints",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload2,
        timeout=30
    )
    results['feature_serving_endpoint'] = {"status": r2.status_code, "body": r2.text[:500]}
except Exception as e:
    results['feature_serving_endpoint'] = {"error": str(e)}

# Check if there's an "online store" API
try:
    r3 = requests.get(
        f"{host}/api/2.0/online-stores",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    results['online_stores'] = {"status": r3.status_code, "body": r3.text[:300]}
except Exception as e:
    results['online_stores'] = {"error": str(e)}

# Check if there's a registered synced tables endpoint through UC
try:
    r4 = requests.get(
        f"{host}/api/2.1/unity-catalog/online-tables",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    results['uc_online_tables'] = {"status": r4.status_code, "body": r4.text[:300]}
except Exception as e:
    results['uc_online_tables'] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(results))
