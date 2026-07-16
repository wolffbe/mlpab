import databricks.sdk as dsdk
w = dsdk.WorkspaceClient()

print("=== WAREHOUSES ===")
for wh in w.warehouses.list():
    print(wh.id, wh.name, wh.state, wh.warehouse_type)

print("=== online_tables methods ===")
print([m for m in dir(w.online_tables) if not m.startswith('_')])

print("=== feature_engineering methods ===")
print([m for m in dir(w.feature_engineering) if not m.startswith('_')])

print("=== feature_store methods ===")
print([m for m in dir(w.feature_store) if not m.startswith('_')])

print("=== jobs submit/create signature hints ===")
import inspect
print([m for m in dir(w.jobs) if not m.startswith('_')])
