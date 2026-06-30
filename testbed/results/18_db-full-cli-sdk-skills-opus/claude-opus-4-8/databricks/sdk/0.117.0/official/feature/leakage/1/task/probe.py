from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
me = w.current_user.me()
print("USER:", me.user_name)
print("=== warehouses ===")
for wh in w.warehouses.list():
    print(wh.id, wh.name, wh.state)
