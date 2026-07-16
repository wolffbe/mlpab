import databricks.sdk as s
import inspect
w = s.WorkspaceClient()
print("=== database methods ===")
for m in dir(w.database):
    if not m.startswith("_"):
        print("  ", m)
import databricks.sdk.service.database as d
print("=== service.database classes ===")
print([x for x in dir(d) if x[0].isupper()])
print("=== SyncedDatabaseTable init ===")
if hasattr(d, "SyncedDatabaseTable"):
    print(inspect.signature(d.SyncedDatabaseTable.__init__))
if hasattr(d, "SyncedTableSpec"):
    print("SPEC:", inspect.signature(d.SyncedTableSpec.__init__))
if hasattr(d, "DatabaseInstance"):
    print("INSTANCE:", inspect.signature(d.DatabaseInstance.__init__))
