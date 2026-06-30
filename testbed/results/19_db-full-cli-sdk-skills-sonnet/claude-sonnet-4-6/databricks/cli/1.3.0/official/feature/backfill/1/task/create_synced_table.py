# Databricks notebook source
# Create synced table for online access

# COMMAND ----------
import requests
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = []

# Create the synced table via REST API with proper body
url = f"https://{host}/api/2.0/postgres/synced_tables"
params = {"synced_table_id": "mlpab0442b8db.mlpab0442b8.accountse81ff1"}
body = {
    "synced_table": {
        "source_table_full_name": "workspace.mlpab0442b8.accountse81ff1"
    }
}
r = requests.post(url, headers=headers, params=params, json=body)
results.append(f"Create synced table: {r.status_code} {r.text[:800]}")

print('\n'.join(results))
spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.synced_table_creation")
