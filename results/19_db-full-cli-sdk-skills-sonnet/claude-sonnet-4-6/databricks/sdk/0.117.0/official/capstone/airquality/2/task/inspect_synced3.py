"""Inspect SyncedDatabaseTable from correct module."""
import os
import inspect
from databricks.sdk.service import database as db_mod

db_attrs = [a for a in dir(db_mod) if not a.startswith("_")]
print("All database module attrs:")
for a in sorted(db_attrs):
    print(f"  {a}")

print()
print("SyncedDatabaseTable source:")
SyncedDatabaseTable = db_mod.SyncedDatabaseTable
print(inspect.getsource(SyncedDatabaseTable))
print()
print("DatabaseInstance source:")
DatabaseInstance = db_mod.DatabaseInstance
print(inspect.getsource(DatabaseInstance)[:2000])
