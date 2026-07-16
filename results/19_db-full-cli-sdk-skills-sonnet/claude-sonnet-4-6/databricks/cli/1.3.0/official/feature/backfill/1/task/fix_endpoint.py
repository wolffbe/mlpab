# Databricks notebook source
# Fix Feature Serving Endpoint

# COMMAND ----------
import requests, json
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = []

# Try to PUT config for the existing endpoint
put_payload = {
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
r = requests.put(f"https://{host}/api/2.0/serving-endpoints/mlpab0442b8-accountse81ff1/config",
    headers=headers, json=put_payload)
results.append(f"PUT config: {r.status_code} {r.text[:500]}")

print('\n'.join(results))

# COMMAND ----------
results2 = []

# Delete and recreate with full spec
r_del = requests.delete(f"https://{host}/api/2.0/serving-endpoints/mlpab0442b8-accountse81ff1", headers=headers)
results2.append(f"Delete: {r_del.status_code}")

# Create with complete spec (workload required?)
create_payload = {
    "name": "mlpab0442b8-accountse81ff1",
    "feature_serving_spec": {
        "feature_lookup_specs": [
            {
                "table_name": "workspace.mlpab0442b8.accountse81ff1",
                "lookup_key": ["row_id"],
                "timestamp_lookup_key": "updated_at"
            }
        ]
    },
    "tags": [{"key": "managed_by", "value": "mlpab0442b8"}]
}
r_create = requests.post(f"https://{host}/api/2.0/serving-endpoints", headers=headers, json=create_payload)
results2.append(f"Create: {r_create.status_code} {r_create.text[:800]}")

print('\n'.join(results2))

# COMMAND ----------
results3 = []

# Check what the endpoint looks like now
r_get = requests.get(f"https://{host}/api/2.0/serving-endpoints/mlpab0442b8-accountse81ff1", headers=headers)
results3.append(f"Get endpoint: {r_get.status_code} {r_get.text}")

# Also try to get the open API spec for serving endpoints to understand the schema
r_oa = requests.get(f"https://{host}/api/2.0/serving-endpoints/mlpab0442b8-accountse81ff1/get-open-api", headers=headers)
results3.append(f"OpenAPI: {r_oa.status_code} {r_oa.text[:200]}")

print('\n'.join(results3))
spark.createDataFrame([(r,) for r in results + results2 + results3], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.endpoint_fix")
