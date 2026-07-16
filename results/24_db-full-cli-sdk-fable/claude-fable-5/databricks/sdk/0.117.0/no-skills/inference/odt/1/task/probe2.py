import inspect
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

w = WorkspaceClient()
print(inspect.signature(w.online_tables.create))
print([a for a in dir(w.online_tables) if not a.startswith('_')])
print([a for a in dir(catalog) if 'Online' in a])
print(inspect.signature(catalog.OnlineTable.__init__))
print(inspect.signature(catalog.OnlineTableSpec.__init__))
