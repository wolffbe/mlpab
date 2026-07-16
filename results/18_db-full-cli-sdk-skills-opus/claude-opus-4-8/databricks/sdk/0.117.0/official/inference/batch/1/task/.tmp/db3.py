import databricks.sdk as s
import databricks.sdk.service.database as d
import inspect
w = s.WorkspaceClient()
print("create_synced_database_table sig:", inspect.signature(w.database.create_synced_database_table))
print("create_database_instance sig:", inspect.signature(w.database.create_database_instance))
print("SchedulingPolicy:", [x for x in dir(d.SyncedTableSchedulingPolicy) if not x.startswith("_")])
print("=== existing instances ===")
for i in w.database.list_database_instances():
    print("  ", i.name, i.state, i.capacity)
