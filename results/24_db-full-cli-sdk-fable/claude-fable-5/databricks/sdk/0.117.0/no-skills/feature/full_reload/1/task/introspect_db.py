import inspect

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.database as db

w = WorkspaceClient()
print([m for m in dir(w.database) if not m.startswith("_")])
print()
print(inspect.signature(w.database.create_database_instance))
print(inspect.signature(w.database.create_synced_database_table))
print()
print([f.name for f in db.DatabaseInstance.__dataclass_fields__.values()])
print()
print([f.name for f in db.SyncedDatabaseTable.__dataclass_fields__.values()])
print([f.name for f in db.SyncedTableSpec.__dataclass_fields__.values()])
print([f.name for f in db.NewPipelineSpec.__dataclass_fields__.values()])
print(list(db.SyncedTableSchedulingPolicy))
