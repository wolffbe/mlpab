"""Inspect Synced Database Tables API and try to create one."""
import os
import inspect
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Inspect create_synced_database_table
print("create_synced_database_table signature:")
sig = inspect.signature(w.database.create_synced_database_table)
print(sig)
print()

# Get source
src = inspect.getsource(w.database.create_synced_database_table)
print("Source:")
print(src)
print()

# Find the SyncedDatabaseTable class
import databricks.sdk.service as svc_pkg
for mod_name in dir(svc_pkg):
    if 'Synced' in mod_name or 'Database' in mod_name:
        print(f"Found: {mod_name}")
