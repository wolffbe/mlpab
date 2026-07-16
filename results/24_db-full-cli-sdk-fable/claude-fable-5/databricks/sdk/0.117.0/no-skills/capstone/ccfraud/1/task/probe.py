import os
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
me = w.current_user.me()
print("user:", me.user_name)
print("schema env:", os.environ["MLPAB_DATABRICKS_SCHEMA"])
print("warehouses:")
for wh in w.warehouses.list():
    print(" ", wh.id, wh.name, wh.state)
