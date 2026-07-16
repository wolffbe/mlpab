import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable,
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)

w = WorkspaceClient()
FQ = "workspace.mlpabce02a9"
NAME = f"{FQ}.transactionsf10ad0_online"

spec = OnlineTableSpec(
    source_table_full_name=f"{FQ}.transactionsf10ad0",
    primary_key_columns=["row_id"],
    timeseries_key="event_time",
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
    perform_full_copy=True,
)
try:
    ot = w.online_tables.create(OnlineTable(name=NAME, spec=spec))
    print("created, waiting for provisioning...")
except Exception as e:
    print("create failed:", e)
    raise

deadline = time.time() + 1500
while time.time() < deadline:
    cur = w.online_tables.get(NAME)
    st = cur.status.detailed_state if cur.status else None
    print("state:", st)
    if st and ("ONLINE" in str(st) and "PIPELINE" not in str(st) or "FAILED" in str(st)):
        break
    time.sleep(20)
print("final:", w.online_tables.get(NAME).status)
