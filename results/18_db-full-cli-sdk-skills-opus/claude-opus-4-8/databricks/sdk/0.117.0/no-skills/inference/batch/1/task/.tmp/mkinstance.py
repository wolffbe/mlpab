from databricks.sdk import WorkspaceClient
import databricks.sdk.service.database as db

w = WorkspaceClient()
name = 'mlpabc69812-scores3380ed-lb'
inst = db.DatabaseInstance(name=name, capacity='CU_1')
res = w.database.create_database_instance_and_wait(inst)
print('instance:', res.name, 'state:', res.state, 'rw_dns:', res.read_write_dns)
