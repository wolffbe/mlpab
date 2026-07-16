import time
import databricks.sdk as dsdk

w = dsdk.WorkspaceClient()
PIPELINE = "705b51a6-98be-420d-9920-d62dcfaa3a8b"
ONLINE_TABLE = "workspace.mlpab2138eb.predictionsa834e5_online"

# poll pipeline until its latest update finishes
deadline = time.monotonic() + 900
last = None
while True:
    p = w.pipelines.get(PIPELINE)
    state = p.state
    if state != last:
        print("pipeline state:", state)
        last = state
    # check latest update
    try:
        updates = w.pipelines.list_updates(pipeline_id=PIPELINE)
        u = updates.updates[0] if updates.updates else None
        if u:
            print("  latest update:", u.update_id, u.state)
            if str(u.state) in ("UpdateInfoState.COMPLETED", "COMPLETED"):
                break
            if str(u.state) in ("UpdateInfoState.FAILED", "FAILED"):
                print("  UPDATE FAILED")
                break
    except Exception as e:
        print("  updates err:", repr(e)[:200])
    if time.monotonic() > deadline:
        print("timeout polling pipeline")
        break
    time.sleep(15)

# verify online table object exists
try:
    t = w.tables.get(ONLINE_TABLE)
    print("online UC table exists:", t.full_name, t.table_type)
except Exception as e:
    print("tables.get err:", repr(e)[:200])
