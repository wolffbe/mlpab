# Databricks notebook source
import requests
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = []
url = f"https://{host}/api/2.0/postgres/synced_tables"
params = {"synced_table_id": "mlpab0442b8db.mlpab0442b8.accountse81ff1"}

# Try different field names
body_variants = [
    {"synced_table": {"delta_table_full_name": "workspace.mlpab0442b8.accountse81ff1"}},
    {"synced_table": {"table_full_name": "workspace.mlpab0442b8.accountse81ff1"}},
    {"synced_table": {"uc_table_full_name": "workspace.mlpab0442b8.accountse81ff1"}},
    {"synced_table": {"source": "workspace.mlpab0442b8.accountse81ff1"}},
    {"synced_table": {"source_full_name": "workspace.mlpab0442b8.accountse81ff1"}},
    {"synced_table": {"table_name": "accountse81ff1", "schema_name": "mlpab0442b8", "catalog_name": "workspace"}},
    {"synced_table": {"delta_catalog": "workspace", "delta_schema": "mlpab0442b8", "delta_table": "accountse81ff1"}},
    {"synced_table": {"sync_status": "ACTIVE", "source_table_full_name": "workspace.mlpab0442b8.accountse81ff1"}},
]

for body in body_variants:
    r = requests.post(url, headers=headers, params=params, json=body)
    field = list(list(body.values())[0].keys())[0]
    results.append(f"Field '{field}': {r.status_code} {r.text[:200]}")

spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.synced_variants")
