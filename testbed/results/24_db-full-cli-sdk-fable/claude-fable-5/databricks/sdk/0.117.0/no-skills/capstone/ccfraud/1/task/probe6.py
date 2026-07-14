from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
for wh in w.warehouses.list():
    print(wh.id, wh.name, wh.state, wh.health)
