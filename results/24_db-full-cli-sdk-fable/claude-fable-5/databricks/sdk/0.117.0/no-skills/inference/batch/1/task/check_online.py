import databricks.sdk
import inspect

sdk = databricks.sdk
db = sdk.service.database
print(inspect.signature(db.DatabaseInstance.__init__))
print()
print(inspect.signature(db.SyncedDatabaseTable.__init__))
print()
print(inspect.signature(db.SyncedTableSpec.__init__))
print()
print(inspect.signature(db.NewPipelineSpec.__init__))
print()
print(list(db.SyncedTableSchedulingPolicy))
print()
w = sdk.WorkspaceClient()
print(inspect.signature(w.database.create_synced_database_table))
print(inspect.signature(w.database.create_database_instance))
