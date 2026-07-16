from databricks.sdk import WorkspaceClient
from databricks.sdk.service import database

w = WorkspaceClient()
print("has database:", hasattr(w, "database"))
print([m for m in dir(w.database) if not m.startswith("_")])
print()
print([c for c in dir(database) if not c.startswith("_")][:60])
print()
help(w.database.create_synced_database_table)
help(database.SyncedTableSpec)
