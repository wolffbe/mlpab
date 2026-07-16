# Databricks notebook source
# Check available SDKs and write results to volume

# COMMAND ----------
import subprocess
results = []
result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'feature' in line.lower() or ('databricks' in line.lower() and 'sdk' in line.lower()):
        results.append(line)

output = '\n'.join(results)
print(output)
dbutils.fs.put("/Volumes/workspace/mlpab0442b8/mlpab0442b8_uploads/sdk_check.txt", output, overwrite=True)

# COMMAND ----------
# Try to use databricks.feature_store (classic FS)
msgs = []
try:
    from databricks.feature_store import FeatureStoreClient
    fs = FeatureStoreClient()
    msgs.append("FeatureStoreClient: available")

    # Get the table
    try:
        ft = fs.get_table("workspace.mlpab0442b8.accountse81ff1")
        msgs.append(f"Table registered: {ft.name}")
        msgs.append(f"PKs: {ft.primary_keys}")
    except Exception as e:
        msgs.append(f"get_table: {e}")

    # Try to publish to online store
    try:
        from databricks.feature_store.online_store_spec import AmazonDynamoDBSpec
        spec = AmazonDynamoDBSpec(
            region="us-east-1",
            write_secret_prefix="",
            read_secret_prefix="",
            table_name="accountse81ff1_online"
        )
        fs.publish_table("workspace.mlpab0442b8.accountse81ff1", spec, mode="overwrite")
        msgs.append("Published to DynamoDB successfully")
    except Exception as e:
        msgs.append(f"DynamoDB publish: {e}")

except Exception as e:
    msgs.append(f"FeatureStoreClient: {e}")

output2 = '\n'.join(msgs)
print(output2)
dbutils.fs.put("/Volumes/workspace/mlpab0442b8/mlpab0442b8_uploads/fs_check.txt", output2, overwrite=True)

# COMMAND ----------
# Try feature-store REST API to register the table
import requests
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# Try registering the table
r = requests.post(f"https://{host}/api/2.0/feature-store/feature-tables",
    headers=headers, json={
        "name": "workspace.mlpab0442b8.accountse81ff1",
        "primary_keys": [{"name": "row_id", "data_type": "string"}],
        "timestamp_keys": [{"name": "updated_at", "data_type": "long"}]
    })
result3 = f"register: {r.status_code} {r.text[:300]}"
print(result3)

# Try create-table endpoint
r2 = requests.post(f"https://{host}/api/2.0/feature-store/create-table",
    headers=headers, json={
        "name": "workspace.mlpab0442b8.accountse81ff1",
        "primary_keys": [{"name": "row_id", "data_type": "string"}],
        "timestamp_keys": [{"name": "updated_at", "data_type": "long"}]
    })
result4 = f"create-table: {r2.status_code} {r2.text[:300]}"
print(result4)

dbutils.fs.put("/Volumes/workspace/mlpab0442b8/mlpab0442b8_uploads/api_check.txt",
    result3 + '\n' + result4, overwrite=True)

# COMMAND ----------
# Try the feature-store feature-specs endpoint
r3 = requests.post(f"https://{host}/api/2.0/feature-store/feature-specs",
    headers=headers, json={
        "name": "workspace.mlpab0442b8.accountse81ff1_spec",
        "features": [{"table_name": "workspace.mlpab0442b8.accountse81ff1", "lookup_key": ["row_id"]}]
    })
result5 = f"feature-specs: {r3.status_code} {r3.text[:500]}"
print(result5)
dbutils.fs.put("/Volumes/workspace/mlpab0442b8/mlpab0442b8_uploads/spec_check.txt", result5, overwrite=True)
