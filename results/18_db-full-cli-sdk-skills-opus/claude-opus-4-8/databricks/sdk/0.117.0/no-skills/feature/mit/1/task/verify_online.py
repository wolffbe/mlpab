import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.pipelines import PipelineState

w = WorkspaceClient()
PIPE = "9c1866bf-fd80-4efe-86b9-c42d87327eed"
ONLINE_TABLE = "workspace.mlpab4bb10d.features74f1ef_online"

# wait for the synced table's pipeline to finish its update
for _ in range(80):
    p = w.pipelines.get(PIPE)
    st = p.state
    # find latest update status
    try:
        updates = w.pipelines.list_pipeline_events(PIPE, max_results=5)
    except Exception:
        pass
    print("pipeline state:", st)
    if st in (PipelineState.IDLE, PipelineState.FAILED):
        break
    time.sleep(15)

# inspect synced table object
try:
    st = w.database.get_synced_database_table(name=ONLINE_TABLE)
    print("synced table status:", st.data_synchronization_status if hasattr(st, "data_synchronization_status") else st)
except Exception as e:
    print("get_synced err:", e)
