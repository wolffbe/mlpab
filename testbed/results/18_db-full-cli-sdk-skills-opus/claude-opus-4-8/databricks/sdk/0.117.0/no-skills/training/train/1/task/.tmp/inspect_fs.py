import inspect
import databricks.sdk as dsdk
w = dsdk.WorkspaceClient()

for name in ["create_online_store", "publish_table", "get_online_store", "list_online_stores"]:
    m = getattr(w.feature_store, name)
    print("== feature_store.", name, "==")
    print(" ", inspect.signature(m))
    print(" ", (m.__doc__ or "").strip()[:600])
    print()

print("=== database methods ===")
print([m for m in dir(w.database) if not m.startswith('_')])
