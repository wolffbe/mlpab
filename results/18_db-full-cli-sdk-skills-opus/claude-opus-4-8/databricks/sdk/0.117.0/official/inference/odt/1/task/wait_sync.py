import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.pipelines import UpdateInfoState

w = WorkspaceClient()
PIPELINE_ID = "a34330d1-5280-40e1-9a18-55750b6a6646"

done = {UpdateInfoState.COMPLETED, UpdateInfoState.FAILED, UpdateInfoState.CANCELED}
for _ in range(80):
    upd = w.pipelines.list_updates(PIPELINE_ID)
    state = upd.updates[0].state if upd.updates else None
    print("update state:", state)
    if state in done:
        break
    time.sleep(15)
print("FINAL update state:", state)
