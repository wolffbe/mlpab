# Databricks notebook source
import requests
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = []

# Use the databricks SDK to look at postgres service
try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    # Check if postgres service exists
    if hasattr(w, 'postgres'):
        results.append(f"postgres service methods: {[m for m in dir(w.postgres) if not m.startswith('_')]}")
        # Try create_synced_table
        import inspect
        sig = inspect.signature(w.postgres.create_synced_table)
        results.append(f"create_synced_table params: {list(sig.parameters.keys())}")
    else:
        results.append("No postgres service in SDK")
except Exception as e:
    results.append(f"SDK error: {e}")

# Try to look at SDK database service for SyncedTable
try:
    import databricks.sdk.service.database as db_svc
    db_attrs = [a for a in dir(db_svc) if not a.startswith('_')]
    results.append(f"database service all: {db_attrs}")

    # Find SyncedTable
    synced_classes = [a for a in db_attrs if 'Synced' in a or 'synced' in a]
    results.append(f"Synced classes: {synced_classes}")
except Exception as e:
    results.append(f"database service: {e}")

spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.postgres_sdk_check")
