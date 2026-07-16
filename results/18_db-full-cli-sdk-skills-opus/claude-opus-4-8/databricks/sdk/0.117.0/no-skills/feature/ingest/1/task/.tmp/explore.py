from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
me = w.current_user.me()
print("USER:", me.user_name)
print("WAREHOUSES:")
for wh in w.warehouses.list():
    print(" ", wh.id, wh.name, str(wh.state), str(wh.warehouse_type))
print("has online_tables:", hasattr(w, "online_tables"))
print("has statement_execution:", hasattr(w, "statement_execution"))
print("has files:", hasattr(w, "files"))
print("has tables:", hasattr(w, "tables"))
