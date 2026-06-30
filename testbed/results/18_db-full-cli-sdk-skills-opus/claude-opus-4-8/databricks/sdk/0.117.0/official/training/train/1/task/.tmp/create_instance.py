import os, datetime
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import DatabaseInstance
w = WorkspaceClient()
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
inst_name = f'{prefix}-lba834e5'

existing = None
for di in w.database.list_database_instances():
    if di.name == inst_name:
        existing = di
print('existing:', existing.state if existing else None)

if existing is None:
    print('creating instance', inst_name)
    res = w.database.create_database_instance_and_wait(
        DatabaseInstance(name=inst_name, capacity='CU_1'),
        timeout=datetime.timedelta(minutes=20))
    print('STATE', res.state, 'rw_dns', res.read_write_dns)
else:
    if str(existing.state) != 'DatabaseInstanceState.AVAILABLE':
        res = w.database.wait_get_database_instance_database_available(
            name=inst_name, timeout=datetime.timedelta(minutes=20))
        print('STATE', res.state)
    else:
        print('already available')
