import inspect
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import database as d
w = WorkspaceClient()
for name in ['create_database_instance', 'create_synced_database_table', 'create_database_catalog']:
    print('===', name, '===')
    print(inspect.signature(getattr(w.database, name)))
for cls in ['DatabaseInstance', 'SyncedDatabaseTable', 'SyncedTableSpec', 'DatabaseCatalog', 'NewPipelineSpec', 'SyncedTableSchedulingPolicy']:
    c = getattr(d, cls)
    if hasattr(c, '__init__') and not issubclass(c, Exception):
        try:
            print('---', cls, inspect.signature(c.__init__))
        except Exception as e:
            print('---', cls, 'enum?', [x for x in dir(c) if not x.startswith('_')][:10])
print('=== existing instances ===')
for di in w.database.list_database_instances():
    print(di.name, di.state)
