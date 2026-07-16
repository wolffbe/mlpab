# Databricks notebook source
# Create Feature Serving Endpoint

# COMMAND ----------
import requests, json
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = []

# Clean up any existing endpoint
r_del = requests.delete(f"https://{host}/api/2.0/serving-endpoints/mlpab0442b8-accountse81ff1", headers=headers)
results.append(f"Delete existing: {r_del.status_code}")

# Try approach 1: feature_serving_spec at top level with workload_spec
payload1 = {
    "name": "mlpab0442b8-accountse81ff1",
    "feature_serving_spec": {
        "feature_lookup_specs": [
            {
                "table_name": "workspace.mlpab0442b8.accountse81ff1",
                "lookup_key": ["row_id"],
                "timestamp_lookup_key": "updated_at"
            }
        ]
    }
}
r1 = requests.post(f"https://{host}/api/2.0/serving-endpoints", headers=headers, json=payload1)
results.append(f"Approach 1 (feature_serving_spec top-level): {r1.status_code} {r1.text[:500]}")

print('\n'.join(results))
spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.endpoint_creation")

# COMMAND ----------
results2 = []

# Try approach 2: with workload spec
if r1.status_code not in [200, 201]:
    payload2 = {
        "name": "mlpab0442b8-accountse81ff1",
        "feature_serving_spec": {
            "feature_lookup_specs": [
                {
                    "table_name": "workspace.mlpab0442b8.accountse81ff1",
                    "lookup_key": ["row_id"],
                    "timestamp_lookup_key": "updated_at"
                }
            ],
            "workload_spec": {
                "workload_size_id": "Small",
                "scale_to_zero_enabled": True
            }
        }
    }
    r2 = requests.post(f"https://{host}/api/2.0/serving-endpoints", headers=headers, json=payload2)
    results2.append(f"Approach 2 (with workload_spec): {r2.status_code} {r2.text[:500]}")

# Try approach 3: using config + feature_serving_spec
    payload3 = {
        "name": "mlpab0442b8-accountse81ff1",
        "config": {
            "feature_serving_spec": {
                "feature_lookup_specs": [
                    {
                        "table_name": "workspace.mlpab0442b8.accountse81ff1",
                        "lookup_key": ["row_id"],
                        "timestamp_lookup_key": "updated_at"
                    }
                ]
            }
        }
    }
    r3 = requests.post(f"https://{host}/api/2.0/serving-endpoints", headers=headers, json=payload3)
    results2.append(f"Approach 3 (config + feature_serving_spec): {r3.status_code} {r3.text[:500]}")

    # Approach 4: check serving-endpoints schema
    r4 = requests.get(f"https://{host}/api/2.0/serving-endpoints/openapi", headers=headers)
    results2.append(f"OpenAPI spec: {r4.status_code} {r4.text[:200]}")

print('\n'.join(results2))
spark.createDataFrame([(r,) for r in results2], ["result"]).write.mode("append").saveAsTable("workspace.mlpab0442b8.endpoint_creation")
