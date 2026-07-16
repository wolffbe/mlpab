import inspect
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
print('db-ish attrs:', [a for a in dir(w) if any(k in a.lower() for k in ('database','synced','lakebase','online'))])
try:
    db = w.database
    print('database api methods:', [m for m in dir(db) if not m.startswith('_')])
except Exception as e:
    print('no w.database:', e)
from databricks.sdk.service import database as dbsvc
print('database service classes:', [n for n in dir(dbsvc) if n[0].isupper()])
