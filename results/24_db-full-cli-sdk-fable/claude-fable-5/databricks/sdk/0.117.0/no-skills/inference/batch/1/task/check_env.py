from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
print("user:", w.current_user.me().user_name)
for wh in w.warehouses.list():
    print("warehouse:", wh.id, wh.name, wh.state)
