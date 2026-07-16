import os
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
me = w.current_user.me()
print("user:", me.user_name)

sch = os.environ["MLPAB_DATABRICKS_SCHEMA"]
try:
    s = w.schemas.get(sch)
    print("schema ok:", s.full_name)
except Exception as e:
    print("schema err:", e)

for wh in w.warehouses.list():
    print("wh:", wh.id, wh.name, wh.state)
