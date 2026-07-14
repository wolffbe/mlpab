# Databricks notebook source
import json, time, requests, os
import mlflow
import mlflow.pyfunc
import pandas as pd

schema = "workspace.mlpabcef85c"
table_name = "profilesaa70e4"
full_table_name = f"{schema}.{table_name}"
prefix = "mlpabcef85c"
endpoint_name = f"{prefix}-feature-serve"
model_name = f"{schema}.{prefix}_feature_model"

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# COMMAND ----------
# Load feature data from Delta table into a dict for the model
print("Loading feature data from Delta table...")
df = spark.table(full_table_name).toPandas()
feature_dict = {}
for _, row in df.iterrows():
    feature_dict[row['account_id']] = [float(row['f1']), float(row['f2']), float(row['f3']), float(row['f4'])]
print(f"Loaded {len(feature_dict)} feature vectors")
print(f"Sample: {list(feature_dict.items())[:2]}")

# COMMAND ----------
# Create a pyfunc model that serves features
class FeatureLookupModel(mlflow.pyfunc.PythonModel):
    def __init__(self, feature_dict):
        self.features = feature_dict

    def predict(self, context, model_input):
        # model_input is a DataFrame with 'account_id' column
        results = []
        for account_id in model_input['account_id']:
            vec = self.features.get(str(account_id), [None, None, None, None])
            results.append({"f1": vec[0], "f2": vec[1], "f3": vec[2], "f4": vec[3]})
        return pd.DataFrame(results)

model = FeatureLookupModel(feature_dict)

# COMMAND ----------
# Set MLflow registry to UC
mlflow.set_registry_uri("databricks-uc")
experiment_path = f"/Users/benedict@logicalclocks.com/{prefix}/feature_model_experiment"
mlflow.set_experiment(experiment_path)

# Register the model
with mlflow.start_run() as run:
    model_info = mlflow.pyfunc.log_model(
        artifact_path="feature_lookup",
        python_model=model,
        registered_model_name=model_name,
        pip_requirements=["mlflow", "pandas"]
    )
    print(f"Model logged: {model_info}")
    print(f"Model URI: {model_info.model_uri}")

# COMMAND ----------
# Get the latest version
from mlflow import MlflowClient
client = MlflowClient()
versions = client.search_model_versions(f"name='{model_name}'")
latest = sorted(versions, key=lambda v: int(v.version))[-1]
model_version = latest.version
print(f"Model version: {model_version}")
model_uri = f"models:/{model_name}/{model_version}"
print(f"Model URI: {model_uri}")

# COMMAND ----------
# Check if serving endpoint already exists
ep_check = requests.get(f"{host}/api/2.0/serving-endpoints/{endpoint_name}", headers=headers)
print(f"Endpoint check: {ep_check.status_code}")

if ep_check.status_code == 200:
    ep_data = ep_check.json()
    ep_state = ep_data.get("state", {}).get("ready", "UNKNOWN")
    print(f"Endpoint exists, state: {ep_state}")
    if ep_state == "READY":
        print("Endpoint already ready!")
else:
    # Create the serving endpoint
    ep_payload = {
        "name": endpoint_name,
        "config": {
            "served_entities": [
                {
                    "name": "feature-lookup-v1",
                    "entity_name": model_name,
                    "entity_version": str(model_version),
                    "workload_size": "Small",
                    "scale_to_zero_enabled": True
                }
            ]
        }
    }
    ep_resp = requests.post(f"{host}/api/2.0/serving-endpoints", headers=headers, json=ep_payload)
    print(f"Endpoint create: {ep_resp.status_code} {ep_resp.text[:500]}")

# COMMAND ----------
# Wait for endpoint to be ready
max_wait = 600
start = time.time()
ep_state = None
while time.time() - start < max_wait:
    poll = requests.get(f"{host}/api/2.0/serving-endpoints/{endpoint_name}", headers=headers)
    if poll.status_code == 200:
        ep_data = poll.json()
        ep_state = ep_data.get("state", {}).get("ready", "NOT_READY")
        print(f"[{int(time.time()-start)}s] Endpoint state: {ep_state}")
        if ep_state == "READY":
            print("Endpoint is READY!")
            break
        if ep_data.get("state", {}).get("config_update") == "UPDATE_FAILED":
            print(f"Endpoint config update FAILED: {json.dumps(ep_data)[:500]}")
            break
    else:
        print(f"Poll error: {poll.status_code}")
        break
    time.sleep(30)

# COMMAND ----------
# Read lookup keys and query the endpoint
with open("/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/lookup_keys.txt") as f:
    lookup_keys = [line.strip() for line in f if line.strip()]
print(f"Querying {len(lookup_keys)} keys via serving endpoint...")

results = {}
for key in lookup_keys:
    query = {
        "dataframe_records": [{"account_id": key}]
    }
    resp = requests.post(
        f"{host}/api/2.0/serving-endpoints/{endpoint_name}/invocations",
        headers=headers,
        json=query
    )
    if resp.status_code == 200:
        data = resp.json()
        if "predictions" in data:
            pred = data["predictions"]
            if isinstance(pred, list) and pred:
                p = pred[0]
                results[key] = [p.get("f1"), p.get("f2"), p.get("f3"), p.get("f4")]
        else:
            print(f"Key {key} unexpected: {json.dumps(data)[:300]}")
    else:
        print(f"Key {key} error: {resp.status_code} {resp.text[:200]}")

print(f"Results count: {len(results)}")

# COMMAND ----------
# Save results
output = {"vectors": results}
with open("/Workspace/Users/benedict@logicalclocks.com/mlpabcef85c/answers.json", "w") as f:
    json.dump(output, f)
print(f"Saved {len(results)} results")
dbutils.notebook.exit(json.dumps({"count": len(results), "sample": dict(list(results.items())[:2])}))
