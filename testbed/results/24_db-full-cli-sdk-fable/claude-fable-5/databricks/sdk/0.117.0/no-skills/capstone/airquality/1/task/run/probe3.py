import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

wh_id = None
for wh in w.warehouses.list():
    print("wh:", wh.id, wh.name, wh.state, flush=True)
    if wh_id is None or "Starter" in (wh.name or ""):
        wh_id = wh.id

print("testing statement execution on", wh_id, flush=True)
try:
    r = w.statement_execution.execute_statement(
        warehouse_id=wh_id, statement="SELECT 1 AS ok", wait_timeout="50s")
    print("status:", r.status.state, "data:", r.result.data_array if r.result else None, flush=True)
except Exception as e:
    print("stmt err:", e, flush=True)

print("trying app create", flush=True)
try:
    from databricks.sdk.service import apps as apps_svc
    app = w.apps.create(app=apps_svc.App(name="mlpab2efe57-airq",
                                         description="airq pipeline runner")).result(timeout=600)
    print("app:", app.name, app.compute_status, app.app_status, flush=True)
    print("sp:", app.service_principal_client_id, app.service_principal_name, app.service_principal_id, flush=True)
except Exception as e:
    print("app err:", type(e).__name__, e, flush=True)
