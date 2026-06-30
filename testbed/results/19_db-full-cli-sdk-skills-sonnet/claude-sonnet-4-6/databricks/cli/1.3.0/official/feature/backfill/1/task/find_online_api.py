# Databricks notebook source
# Find the correct API endpoint for Synced Tables / online feature serving

# COMMAND ----------
import requests

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = []

# Try different API paths and methods
api_paths = [
    ("GET", "/api/2.0/online-tables"),
    ("GET", "/api/2.0/synced-tables"),
    ("GET", "/api/2.1/online-tables"),
    ("GET", "/api/2.0/unitycatalog/online-tables"),
    ("GET", "/api/2.0/uc/online-tables"),
    ("GET", "/api/preview/online-tables"),
    ("GET", "/api/preview/synced-tables"),
    ("GET", "/api/2.0/feature-store/online-stores"),
]

for method, path in api_paths:
    try:
        if method == "GET":
            r = requests.get(f"https://{host}{path}", headers=headers)
        else:
            r = requests.post(f"https://{host}{path}", headers=headers, json={})
        results.append(f"{method} {path}: {r.status_code} {r.text[:100]}")
    except Exception as e:
        results.append(f"{method} {path}: ERROR {e}")

output = '\n'.join(results)
print(output)
spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.api_discovery")

# COMMAND ----------
# Try creating synced table via databricks SDK
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
results2 = []

# Try to create online table via SDK (bypasses CLI validation)
try:
    from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTable
    spec = OnlineTableSpec(
        source_table_full_name="workspace.mlpab0442b8.accountse81ff1",
        primary_key_columns=["row_id", "updated_at"],
        run_triggered={}
    )
    ot = w.online_tables.create(
        name="workspace.mlpab0442b8.accountse81ff1_online",
        spec=spec
    )
    results2.append(f"Online table created: {ot}")
except Exception as e:
    results2.append(f"Online table SDK: {e}")

# Try other SDK methods
try:
    methods = [m for m in dir(w) if 'online' in m.lower() or 'sync' in m.lower()]
    results2.append(f"SDK methods with online/sync: {methods}")
except Exception as e:
    results2.append(f"SDK dir error: {e}")

output2 = '\n'.join(results2)
print(output2)
spark.createDataFrame([(r,) for r in results2], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.sdk_discovery")
