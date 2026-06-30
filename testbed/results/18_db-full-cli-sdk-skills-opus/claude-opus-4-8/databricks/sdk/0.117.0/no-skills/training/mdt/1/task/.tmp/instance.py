import datetime
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.database as db

w = WorkspaceClient()
INST = 'mlpabc1d5e2-scaled'
inst = db.DatabaseInstance(name=INST, capacity='CU_1')
res = w.database.create_database_instance_and_wait(inst, timeout=datetime.timedelta(minutes=25))
print('instance:', res.name, 'state:', res.state)
