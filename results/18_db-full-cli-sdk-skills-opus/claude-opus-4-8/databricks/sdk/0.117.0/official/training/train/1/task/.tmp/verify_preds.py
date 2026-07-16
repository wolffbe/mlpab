import os, inspect
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
cat, sch = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
vol_path = f'/Volumes/{cat}/{sch}/trainvola834e5'
resp = w.files.download(f'{vol_path}/predictions.csv')
data = resp.contents.read().decode()
lines = data.splitlines()
print('rows:', len(lines))
print('\n'.join(lines[:4]))
print('...')
print(lines[-1])

from databricks.sdk.service import catalog
print('=== OnlineTableSpec params ===')
print([n for n in dir(catalog) if 'Online' in n])
print(inspect.signature(catalog.OnlineTableSpec.__init__))
