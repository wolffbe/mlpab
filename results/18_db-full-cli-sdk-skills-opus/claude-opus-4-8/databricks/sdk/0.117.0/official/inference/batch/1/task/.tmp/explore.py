import databricks.sdk as s
w = s.WorkspaceClient()
me = w.current_user.me()
print("USER:", me.user_name)
print("WAREHOUSES:")
for wh in w.warehouses.list():
    print("  ", wh.id, wh.name, wh.state)
print("HAS online_tables:", hasattr(w, "online_tables"))
print("HAS feature_store:", hasattr(w, "feature_store"))
print("HAS statement_execution:", hasattr(w, "statement_execution"))
print("HAS files:", hasattr(w, "files"))
print("HAS tables:", hasattr(w, "tables"))
