# Databricks notebook source
# Check endpoint config and try SDK approach

# COMMAND ----------
import requests, json
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = []

# Get endpoint config in full detail
r = requests.get(f"https://{host}/api/2.0/serving-endpoints/mlpab0442b8-accountse81ff1", headers=headers)
results.append(f"Full endpoint: {r.status_code} {r.text}")

print('\n'.join(results))

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, CreateServingEndpoint
import inspect

w = WorkspaceClient()
results2 = []

# Check CreateServingEndpoint fields
results2.append(f"CreateServingEndpoint fields: {[f for f in dir(CreateServingEndpoint) if not f.startswith('_')]}")

# Check ServedEntityInput
results2.append(f"ServedEntityInput workload fields: {[f for f in dir(ServedEntityInput) if 'work' in f.lower() or 'feature' in f.lower()]}")

# Try creating with SDK including feature_serving_spec
try:
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

    # Check if EndpointCoreConfigInput has feature_serving_spec
    csc = EndpointCoreConfigInput(served_entities=[])
    results2.append(f"EndpointCoreConfigInput fields: {list(csc.as_dict().keys())}")
    results2.append(f"All EndpointCoreConfigInput attrs: {[f for f in dir(csc) if not f.startswith('_')]}")
except Exception as e:
    results2.append(f"Error: {e}")

print('\n'.join(results2))
spark.createDataFrame([(r,) for r in results + results2], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.endpoint_check")
