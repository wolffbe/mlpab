from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

w = WorkspaceClient()
print("has online_tables:", hasattr(w, "online_tables"))
print([m for m in dir(w.online_tables) if not m.startswith("_")])
print([c for c in dir(catalog) if "Online" in c])
help(w.online_tables.create)
