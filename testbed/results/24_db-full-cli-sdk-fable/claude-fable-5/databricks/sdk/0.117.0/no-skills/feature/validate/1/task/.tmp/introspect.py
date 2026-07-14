import inspect

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.database as db

w = WorkspaceClient()
print([m for m in dir(w.database) if not m.startswith("_")])
print()
print(inspect.signature(w.database.create_database_instance))
print(inspect.signature(w.database.create_synced_database_table))
print()
print("SyncedDatabaseTable fields:", [f for f in db.SyncedDatabaseTable.__dataclass_fields__])
print("SyncedTableSpec fields:", [f for f in db.SyncedTableSpec.__dataclass_fields__])
print("DatabaseInstance fields:", [f for f in db.DatabaseInstance.__dataclass_fields__])
print("SchedulingPolicy:", [x for x in dir(db.SyncedTableSchedulingPolicy) if not x.startswith("_")])
print("NewPipelineSpec fields:", [f for f in db.NewPipelineSpec.__dataclass_fields__])
