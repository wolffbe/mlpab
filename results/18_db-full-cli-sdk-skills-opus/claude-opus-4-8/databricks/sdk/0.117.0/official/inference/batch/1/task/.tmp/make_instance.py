import databricks.sdk as s
import databricks.sdk.service.database as d
import os

w = s.WorkspaceClient()
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
inst_name = f"{prefix}-lakebase"

inst = d.DatabaseInstance(name=inst_name, capacity="CU_1")
print("creating instance", inst_name)
waiter = w.database.create_database_instance(inst)
res = waiter.result(timeout=__import__("datetime").timedelta(minutes=20))
print("instance state:", res.state, "name:", res.name)
