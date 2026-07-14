from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
print("user:", w.current_user.me().user_name)
for wh in w.warehouses.list():
    print("warehouse:", wh.id, wh.name, wh.state)
print("online/feature APIs:", [a for a in dir(w) if 'online' in a.lower() or 'feature' in a.lower() or 'synced' in a.lower()])
