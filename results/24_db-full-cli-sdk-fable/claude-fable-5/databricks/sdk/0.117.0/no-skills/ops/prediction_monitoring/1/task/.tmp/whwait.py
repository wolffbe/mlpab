import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import State

w = WorkspaceClient()
for x in w.warehouses.list():
    print(x.name, x.id, x.state, "type:", x.warehouse_type, "serverless:", x.enable_serverless_compute, "health:", x.health)

t0 = time.time()
while time.time() - t0 < 1500:
    states = {x.name: x.state for x in w.warehouses.list()}
    if any(s == State.RUNNING for s in states.values()):
        print("RUNNING:", states)
        break
    time.sleep(20)
else:
    print("timeout, last states:", states)
