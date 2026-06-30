# Databricks notebook source
# Try creating proper Feature Serving Endpoint

# COMMAND ----------
import requests, json
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = []

# Delete current endpoint
r_del = requests.delete(f"https://{host}/api/2.0/serving-endpoints/mlpab0442b8-accountse81ff1", headers=headers)
results.append(f"Deleted: {r_del.status_code}")

# Try creating with SDK - ServedEntityInput can create feature serving
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, CreateServingEndpoint
import inspect

# Check SDK class CreateServingEndpoint
w = WorkspaceClient()
results.append(f"CreateServingEndpoint attrs: {[f for f in dir(CreateServingEndpoint) if not f.startswith('_')]}")

# Try creating with SDK
try:
    # Build the request as dict first to see what fields are available
    cse = CreateServingEndpoint(name="mlpab0442b8-accountse81ff1")
    results.append(f"CreateServingEndpoint dict: {cse.as_dict()}")
    results.append(f"CreateServingEndpoint all fields: {list(inspect.signature(CreateServingEndpoint.__init__).parameters.keys())}")
except Exception as e:
    results.append(f"SDK error: {e}")

print('\n'.join(results))
spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.endpoint_try2")

# COMMAND ----------
results2 = []

# Check if there's a feature_serving_spec equivalent in SDK
try:
    from databricks.sdk.service.serving import (
        CreateServingEndpoint, EndpointCoreConfigInput, ServedEntityInput
    )
    params = list(inspect.signature(CreateServingEndpoint.__init__).parameters.keys())
    results2.append(f"CreateServingEndpoint init params: {params}")

    # Try with feature_serving_spec as keyword argument
    cse = CreateServingEndpoint(
        name="test",
        feature_serving_spec={}  # This might fail with unknown kwarg
    )
    results2.append(f"feature_serving_spec accepted: {cse.as_dict()}")
except Exception as e:
    results2.append(f"feature_serving_spec test: {e}")

print('\n'.join(results2))
spark.createDataFrame([(r,) for r in results2], ["result"]).write.mode("append").saveAsTable("workspace.mlpab0442b8.endpoint_try2")

# COMMAND ----------
results3 = []

# Try the SDK's create with feature_serving_spec using as_dict approach
try:
    from databricks.sdk.service.serving import CreateServingEndpoint

    # Create using as_dict and then raw API
    payload_dict = {
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

    # Use the SDK's HTTP client directly
    w._api_client.do("POST", "/api/2.0/serving-endpoints", body=payload_dict)
    results3.append("Created via SDK HTTP client!")
except Exception as e:
    results3.append(f"SDK HTTP error: {e}")

# Also check if there's a feature-specs API
for path in ["/api/2.0/feature-store/feature-specs", "/api/2.0/feature-specs"]:
    r = requests.post(f"https://{host}{path}", headers=headers, json={
        "name": "workspace.mlpab0442b8.accountse81ff1_spec",
        "features": [{"table_name": "workspace.mlpab0442b8.accountse81ff1", "lookup_key": ["row_id"]}]
    })
    results3.append(f"POST {path}: {r.status_code} {r.text[:300]}")

print('\n'.join(results3))
spark.createDataFrame([(r,) for r in results3], ["result"]).write.mode("append").saveAsTable("workspace.mlpab0442b8.endpoint_try2")
