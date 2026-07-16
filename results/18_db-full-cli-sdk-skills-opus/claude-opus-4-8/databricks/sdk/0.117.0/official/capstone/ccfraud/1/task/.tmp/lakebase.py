from databricks.sdk import WorkspaceClient
from databricks.sdk.service import database
import os, datetime

w = WorkspaceClient()
PREFIX = os.environ['MLPAB_DATABRICKS_PREFIX']
inst_name = f"{PREFIX}-ccfraud"

# 1) create / get database instance
try:
    inst = w.database.get_database_instance(inst_name)
    print("instance exists, state", inst.state, flush=True)
except Exception:
    print("creating instance", inst_name, flush=True)
    wait = w.database.create_database_instance(
        database.DatabaseInstance(name=inst_name, capacity="CU_1")
    )
    inst = wait.result(timeout=datetime.timedelta(seconds=1200))
    print("instance state", inst.state, flush=True)
