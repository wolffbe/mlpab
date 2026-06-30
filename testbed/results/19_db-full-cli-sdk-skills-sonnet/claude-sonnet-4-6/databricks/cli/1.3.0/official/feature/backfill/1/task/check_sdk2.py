# Databricks notebook source
# Check available SDKs - write output to UC table

# COMMAND ----------
import subprocess
results = []
result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'feature' in line.lower() or ('databricks' in line.lower()):
        results.append(line)

output = '\n'.join(results)
print(output)

# Write to table for easy reading
spark.createDataFrame([(r,) for r in results], ["line"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.sdk_check_output")

# COMMAND ----------
# Try to use databricks.feature_store (classic FS)
msgs = []
try:
    from databricks.feature_store import FeatureStoreClient
    fs = FeatureStoreClient()
    msgs.append("FeatureStoreClient: available")
except Exception as e:
    msgs.append(f"FeatureStoreClient error: {e}")

# Check for feature engineering
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    msgs.append("FeatureEngineeringClient: available")
except Exception as e:
    msgs.append(f"FeatureEngineeringClient: {e}")

# Check REST API for online stores
import requests
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Try publishing UC table to online store using REST
payload = {
    "publish_spec": {
        "online_store": "mlpab0442b8-online-store",
        "online_table_name": "accountse81ff1",
        "publish_mode": 1
    }
}
r = requests.post(f"https://{host}/api/2.0/feature-store/tables/workspace.mlpab0442b8.accountse81ff1/publish",
    headers=headers, json=payload)
msgs.append(f"Publish response: {r.status_code} {r.text[:300]}")

# Try with mode 2
payload2 = {
    "publish_spec": {
        "online_store": "mlpab0442b8-online-store",
        "online_table_name": "accountse81ff1",
        "publish_mode": 2
    }
}
r2 = requests.post(f"https://{host}/api/2.0/feature-store/tables/workspace.mlpab0442b8.accountse81ff1/publish",
    headers=headers, json=payload2)
msgs.append(f"Publish mode 2: {r2.status_code} {r2.text[:300]}")

output2 = '\n'.join(msgs)
print(output2)
spark.createDataFrame([(m,) for m in msgs], ["message"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.api_check_output")
