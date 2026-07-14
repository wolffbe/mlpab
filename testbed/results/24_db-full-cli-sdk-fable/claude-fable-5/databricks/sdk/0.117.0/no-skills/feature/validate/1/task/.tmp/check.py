import databricks.sdk
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
print("me:", w.current_user.me().user_name)
for wh in w.warehouses.list():
    print("warehouse:", wh.id, wh.name, wh.state)
print("has online_tables:", hasattr(w, "online_tables"))
print("has database:", hasattr(w, "database"))
print("has statement_execution:", hasattr(w, "statement_execution"))
