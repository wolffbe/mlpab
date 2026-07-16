"""Inspect SyncedDatabaseTable class and DatabaseInstance."""
import os
import inspect
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import SyncedDatabaseTable

w = WorkspaceClient()

print("SyncedDatabaseTable source:")
try:
    print(inspect.getsource(SyncedDatabaseTable))
except Exception as e:
    print(f"catalog import failed: {e}")

# Try other locations
from databricks.sdk.service import database as db_svc
print("database service module attrs:")
db_attrs = [a for a in dir(db_svc) if 'Synced' in a or 'Database' in a or 'Instance' in a]
print(db_attrs[:30])
print()

# Get the SyncedDatabaseTable from the module
for attr in db_attrs:
    if 'Synced' in attr:
        cls = getattr(db_svc, attr)
        try:
            src = inspect.getsource(cls)
            print(f"{attr}:")
            print(src[:2000])
            print()
        except:
            pass
