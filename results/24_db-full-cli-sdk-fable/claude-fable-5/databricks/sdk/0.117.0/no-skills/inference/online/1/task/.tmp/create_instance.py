import os

import databricks.sdk
import databricks.sdk.service.database as db

w = databricks.sdk.WorkspaceClient()
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
inst_name = f"{prefix}-online"

try:
    inst = w.database.get_database_instance(inst_name)
    print("exists:", inst.name, inst.state)
except Exception as e:
    print("creating:", inst_name)
    inst = w.database.create_database_instance_and_wait(
        db.DatabaseInstance(name=inst_name, capacity="CU_1")
    )
print("instance:", inst.name, inst.state, inst.read_write_dns)
