import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
PID = "b88cff41-1254-4ae6-8a71-b44429946751"

deadline = time.time() + 1500
attempt = 0
while time.time() < deadline:
    attempt += 1
    try:
        upd = w.pipelines.start_update(pipeline_id=PID, full_refresh=True)
    except Exception as e:
        print(f"start attempt {attempt}: {e}", flush=True)
        time.sleep(45)
        continue
    print(f"attempt {attempt} update:", upd.update_id, flush=True)
    while True:
        u = w.pipelines.get_update(pipeline_id=PID, update_id=upd.update_id).update
        s = str(u.state)
        if s in ("UpdateInfoState.COMPLETED", "UpdateInfoState.FAILED",
                 "UpdateInfoState.CANCELED"):
            break
        time.sleep(20)
    print("  ->", s, flush=True)
    if s == "UpdateInfoState.COMPLETED":
        print("PIPELINE COMPLETED", flush=True)
        raise SystemExit(0)
    for ev in w.pipelines.list_pipeline_events(pipeline_id=PID, max_results=25):
        if ev.level and str(ev.level) == "EventLevel.ERROR":
            print("  err:", (ev.message or "")[:300], flush=True)
            if ev.error:
                for ex in ev.error.exceptions or []:
                    print("    ex:", (ex.message or "")[:500], flush=True)
            break
    time.sleep(45)
raise SystemExit("gave up after deadline")
