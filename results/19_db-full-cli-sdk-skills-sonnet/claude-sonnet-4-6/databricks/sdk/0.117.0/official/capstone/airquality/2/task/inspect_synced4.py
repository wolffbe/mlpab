"""Inspect SyncedTableSpec and create Lakebase synced table."""
import os
import inspect
from databricks.sdk.service.database import SyncedTableSpec, SyncedTableSpec

print("SyncedTableSpec source:")
print(inspect.getsource(SyncedTableSpec))
