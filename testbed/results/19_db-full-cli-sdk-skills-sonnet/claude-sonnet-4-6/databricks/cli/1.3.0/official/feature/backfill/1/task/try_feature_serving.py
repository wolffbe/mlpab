# Databricks notebook source
# Try creating a Feature Serving Endpoint

# COMMAND ----------
import requests, json
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = []

# First, look at the serving endpoint schema
r = requests.get(f"https://{host}/api/2.0/serving-endpoints", headers=headers)
results.append(f"GET serving-endpoints: {r.status_code} {r.text[:200]}")

# Try feature serving spec approaches
# Approach 1: Use feature_spec with served_entities
payload1 = {
    "name": "mlpab0442b8-feat-serving",
    "config": {
        "served_entities": [
            {
                "entity_name": "workspace.mlpab0442b8.accountse81ff1",
                "entity_version": "1",
                "feature_serving_spec": {
                    "feature_lookup_spec": [
                        {
                            "table_name": "workspace.mlpab0442b8.accountse81ff1",
                            "lookup_key": ["row_id"],
                            "timestamp_lookup_key": "updated_at"
                        }
                    ]
                }
            }
        ]
    }
}
r1 = requests.post(f"https://{host}/api/2.0/serving-endpoints", headers=headers, json=payload1)
results.append(f"Feature serving attempt 1: {r1.status_code} {r1.text[:400]}")

print('\n'.join(results))
spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.serving_try1")

# COMMAND ----------
results2 = []

# Approach 2: Use the SDK to check feature spec support
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import serving

w = WorkspaceClient()

# Check if there are feature serving specific methods
methods = [m for m in dir(w.serving_endpoints) if not m.startswith('_')]
results2.append(f"serving_endpoints methods: {methods}")

# Check serving models
try:
    from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput
    se = ServedEntityInput(
        entity_name="workspace.mlpab0442b8.accountse81ff1",
        entity_version="1"
    )
    results2.append(f"ServedEntityInput fields: {[f for f in se.__dict__ if not f.startswith('_')]}")
    results2.append(f"ServedEntityInput help: {[f for f in dir(ServedEntityInput) if not f.startswith('_')]}")
except Exception as e:
    results2.append(f"ServedEntityInput: {e}")

print('\n'.join(results2))
spark.createDataFrame([(r,) for r in results2], ["result"]).write.mode("append").saveAsTable("workspace.mlpab0442b8.serving_try1")

# COMMAND ----------
results3 = []

# Try feature specs API
r4 = requests.post(f"https://{host}/api/2.0/feature-store/feature-specs",
    headers=headers,
    json={
        "name": "workspace.mlpab0442b8.accountse81ff1_spec",
        "features": [
            {
                "table_name": "workspace.mlpab0442b8.accountse81ff1",
                "lookup_key": ["row_id"],
                "timestamp_lookup_key": "updated_at"
            }
        ]
    }
)
results3.append(f"POST feature-specs: {r4.status_code} {r4.text[:400]}")

# Also check what feature-store API paths exist
for path in ["/api/2.0/feature-store/feature-specs", "/api/2.0/feature-store/tables"]:
    r = requests.get(f"https://{host}{path}", headers=headers)
    results3.append(f"GET {path}: {r.status_code} {r.text[:200]}")

print('\n'.join(results3))
spark.createDataFrame([(r,) for r in results3], ["result"]).write.mode("append").saveAsTable("workspace.mlpab0442b8.serving_try1")
