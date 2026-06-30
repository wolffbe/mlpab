from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as cat

w = WorkspaceClient()
print('online_tables methods:', [m for m in dir(w.online_tables) if not m.startswith('_')])
print()
import inspect
print('create sig:', inspect.signature(w.online_tables.create))
print()
# look for spec classes
names = [n for n in dir(cat) if 'Online' in n or 'Spec' in n]
print('catalog spec classes:', names)
for n in ['OnlineTable', 'OnlineTableSpec', 'OnlineTableSpecTriggeredSchedulingPolicy']:
    if hasattr(cat, n):
        c = getattr(cat, n)
        try:
            print(n, inspect.signature(c.__init__))
        except Exception as e:
            print(n, 'no sig', e)
