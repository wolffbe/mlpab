# Databricks notebook source
import requests, inspect
import databricks.sdk.service.database as db_svc

host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = []

# COMMAND ----------
# Check DatabaseAPI methods for synced table creation
try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    # Check if there's a database API
    if hasattr(w, 'database'):
        db_methods = [m for m in dir(w.database) if not m.startswith('_')]
        results.append(f"database methods: {db_methods}")
    else:
        results.append("No database service in w")
        # Try creating DatabaseAPI directly
        api = db_svc.DatabaseAPI(w._api_client)
        db_methods2 = [m for m in dir(api) if not m.startswith('_')]
        results.append(f"DatabaseAPI methods: {db_methods2}")
except Exception as e:
    results.append(f"DatabaseAPI: {e}")

spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.synced_v2_output")

# COMMAND ----------
results2 = []

# Try with correct nested structure
url = f"https://{host}/api/2.0/postgres/synced_tables"
params = {"synced_table_id": "mlpab0442b8db.mlpab0442b8.accountse81ff1"}
body = {
    "synced_table": {
        "spec": {
            "source_table_full_name": "workspace.mlpab0442b8.accountse81ff1",
            "primary_key_columns": ["row_id", "updated_at"],
            "timeseries_key": "updated_at",
            "create_database_objects_if_missing": True,
            "scheduling_policy": "TRIGGERED"
        }
    }
}
r = requests.post(url, headers=headers, params=params, json=body)
results2.append(f"Create synced (with spec): {r.status_code} {r.text[:600]}")

spark.createDataFrame([(r,) for r in results2], ["result"]).write.mode("append").saveAsTable("workspace.mlpab0442b8.synced_v2_output")

# COMMAND ----------
results3 = []

# Also check what API the DatabaseAPI uses
try:
    api2 = db_svc.DatabaseAPI(w._api_client)
    # Check if it has synced table creation
    st_methods = [m for m in dir(api2) if 'synced' in m.lower() or 'table' in m.lower()]
    results3.append(f"DatabaseAPI table methods: {st_methods}")

    # Try create_synced_database_table
    if hasattr(api2, 'create_synced_database_table'):
        sig = inspect.signature(api2.create_synced_database_table)
        results3.append(f"create_synced_database_table sig: {list(sig.parameters.keys())}")
except Exception as e:
    results3.append(f"DatabaseAPI methods: {e}")

spark.createDataFrame([(r,) for r in results3], ["result"]).write.mode("append").saveAsTable("workspace.mlpab0442b8.synced_v2_output")
