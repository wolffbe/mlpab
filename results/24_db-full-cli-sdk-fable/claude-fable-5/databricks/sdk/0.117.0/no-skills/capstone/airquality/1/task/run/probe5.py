import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import apps as apps_svc

w = WorkspaceClient()
PID = "b88cff41-1254-4ae6-8a71-b44429946751"
WHS = ["a832b544eb7dc3fe", "8a93fc195da2ceb1"]

deadline = time.time() + 480
cycle = 0
winner = None
while time.time() < deadline and winner is None:
    cycle += 1
    # 1) DLT
    try:
        upd = w.pipelines.start_update(pipeline_id=PID, full_refresh=True)
        for _ in range(6):
            u = w.pipelines.get_update(pipeline_id=PID, update_id=upd.update_id).update
            s = str(u.state)
            if s in ("UpdateInfoState.FAILED", "UpdateInfoState.CANCELED"):
                break
            if s not in ("UpdateInfoState.CREATED", "UpdateInfoState.QUEUED",
                         "UpdateInfoState.WAITING_FOR_RESOURCES", "UpdateInfoState.INITIALIZING"):
                winner = ("dlt", upd.update_id)
                break
            time.sleep(15)
        print(f"c{cycle} dlt:", s, flush=True)
        if s not in ("UpdateInfoState.FAILED", "UpdateInfoState.CANCELED") and winner is None:
            winner = ("dlt", upd.update_id)
    except Exception as e:
        print(f"c{cycle} dlt exc:", str(e)[:120], flush=True)
    if winner:
        break
    # 2) warehouses
    for wh in WHS:
        try:
            r = w.statement_execution.execute_statement(
                warehouse_id=wh, statement="SELECT 1", wait_timeout="30s")
            s = str(r.status.state)
            print(f"c{cycle} wh {wh}:", s, flush=True)
            if s == "StatementState.SUCCEEDED":
                winner = ("wh", wh)
                break
        except Exception as e:
            print(f"c{cycle} wh {wh} exc:", str(e)[:120], flush=True)
    if winner:
        break
    # 3) app
    try:
        aw = w.apps.create(app=apps_svc.App(name="mlpab2efe57-airq", description="airq runner"))
        print(f"c{cycle} app create accepted", flush=True)
        winner = ("app", "mlpab2efe57-airq")
        break
    except Exception as e:
        print(f"c{cycle} app exc:", str(e)[:100], flush=True)
    time.sleep(20)

print("WINNER:", winner, flush=True)
