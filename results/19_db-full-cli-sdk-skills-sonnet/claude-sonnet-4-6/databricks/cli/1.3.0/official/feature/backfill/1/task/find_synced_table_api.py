# Databricks notebook source
# Find the correct API for synced tables

# COMMAND ----------
import requests, inspect
host = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

results = []

# Check if there's a SyncedTable class in the SDK
from databricks.sdk.service import database as db_service
results.append(f"database service attrs: {[a for a in dir(db_service) if 'synced' in a.lower() or 'table' in a.lower()]}")

# Check all classes
all_classes = [a for a in dir(db_service) if not a.startswith('_')]
results.append(f"all attrs: {all_classes[:50]}")

print('\n'.join(results))
spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.synced_api_discovery")

# COMMAND ----------
results2 = []

# Try finding SyncedTable in SDK
try:
    from databricks.sdk.service.database import SyncedTable
    sig = inspect.signature(SyncedTable.__init__)
    results2.append(f"SyncedTable init params: {list(sig.parameters.keys())}")
    # Create instance to see default dict
    st = SyncedTable()
    results2.append(f"SyncedTable dict: {st.as_dict()}")
except Exception as e:
    results2.append(f"SyncedTable error: {e}")

# Try PostgresAPI
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
pg_methods = [m for m in dir(w.postgres) if not m.startswith('_')]
results2.append(f"postgres methods: {pg_methods}")

# Try create_synced_table via SDK
try:
    sig2 = inspect.signature(w.postgres.create_synced_table)
    results2.append(f"create_synced_table params: {list(sig2.parameters.keys())}")
except Exception as e:
    results2.append(f"create_synced_table sig: {e}")

print('\n'.join(results2))
spark.createDataFrame([(r,) for r in results2], ["result"]).write.mode("append").saveAsTable("workspace.mlpab0442b8.synced_api_discovery")

# COMMAND ----------
results3 = []

# Try using SDK to create synced table
from databricks.sdk.service.database import SyncedTable, CreateSyncedTableRequest

try:
    sig3 = inspect.signature(CreateSyncedTableRequest.__init__)
    results3.append(f"CreateSyncedTableRequest params: {list(sig3.parameters.keys())}")
    cstr = CreateSyncedTableRequest()
    results3.append(f"CreateSyncedTableRequest dict: {cstr.as_dict()}")
except Exception as e:
    results3.append(f"CreateSyncedTableRequest: {e}")

print('\n'.join(results3))
spark.createDataFrame([(r,) for r in results3], ["result"]).write.mode("append").saveAsTable("workspace.mlpab0442b8.synced_api_discovery")
