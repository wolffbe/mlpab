from databricks.sdk import WorkspaceClient
import inspect
w = WorkspaceClient()
print('database methods:', [m for m in dir(w.database) if not m.startswith('_')])
import databricks.sdk.service.database as db
print()
print('module classes:', [n for n in dir(db) if n[0].isupper()])
for m in ['create_database_instance', 'create_synced_database_table', 'create_database_catalog']:
    if hasattr(w.database, m):
        print(m, inspect.signature(getattr(w.database, m)))
