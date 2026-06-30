from databricks.sdk import WorkspaceClient
from databricks.sdk.service import database as db
from datetime import timedelta
w = WorkspaceClient()
PREFIX = 'mlpab6d0586'
C = 'workspace.mlpab6d0586'
INST = f"{PREFIX}-lakebase"

inst = w.database.get_database_instance(name=INST) if any(
    i.name == INST for i in w.database.list_database_instances()) else None
if inst is None:
    print('creating instance', INST)
    inst = w.database.create_database_instance_and_wait(
        db.DatabaseInstance(name=INST, capacity='CU_1'), timeout=timedelta(minutes=20))
print('instance state:', inst.state, 'name:', inst.name)
