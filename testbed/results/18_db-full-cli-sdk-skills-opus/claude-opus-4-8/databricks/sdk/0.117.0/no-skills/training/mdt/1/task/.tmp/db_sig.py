import inspect
import databricks.sdk.service.database as db
for n in ['DatabaseInstance', 'SyncedDatabaseTable', 'SyncedTableSpec', 'SyncedTableSchedulingPolicy', 'DatabaseCatalog']:
    c = getattr(db, n)
    try:
        print(n, inspect.signature(c.__init__))
    except Exception as e:
        print(n, e)
    print()
print('SyncedTableSchedulingPolicy members:', [x for x in dir(db.SyncedTableSchedulingPolicy) if not x.startswith('_')])
