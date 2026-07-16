import inspect

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import database

w = WorkspaceClient()
print([m for m in dir(w.database) if not m.startswith("_")])
print()
print(inspect.signature(w.database.create_database_instance))
print(inspect.signature(w.database.create_synced_database_table))
print()
print(inspect.getsource(database.SyncedTableSpec))
print()
print(inspect.getsource(database.SyncedDatabaseTable)[:2000])
print()
print(inspect.getsource(database.NewPipelineSpec)[:1500])
print()
print([f.name for f in database.DatabaseInstance.__dataclass_fields__.values()] if hasattr(database.DatabaseInstance, "__dataclass_fields__") else inspect.signature(database.DatabaseInstance.__init__))
