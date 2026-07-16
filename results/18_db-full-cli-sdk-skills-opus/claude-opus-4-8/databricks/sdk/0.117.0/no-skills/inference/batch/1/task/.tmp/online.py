from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
print('online_tables methods:', [m for m in dir(w.online_tables) if not m.startswith('_')])
import databricks.sdk.service.catalog as cat
print([n for n in dir(cat) if 'Online' in n or 'Spec' in n])
