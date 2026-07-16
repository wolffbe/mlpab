from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
me = w.current_user.me()
print("user:", me.user_name)
for wh in w.warehouses.list():
    print("warehouse:", wh.id, wh.name, wh.state)
print("has online_tables:", hasattr(w, "online_tables"))
print("has database:", hasattr(w, "database"))
