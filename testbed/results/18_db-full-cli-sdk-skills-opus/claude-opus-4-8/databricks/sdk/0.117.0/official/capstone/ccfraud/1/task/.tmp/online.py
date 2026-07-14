from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog
import datetime

w = WorkspaceClient()
SCHEMA = "workspace.mlpabea3b07"
src = f"{SCHEMA}.ccpred2dbe0a"
online_name = f"{SCHEMA}.ccpred2dbe0a_online"

spec = catalog.OnlineTableSpec(
    source_table_full_name=src,
    primary_key_columns=["transaction_id"],
    run_triggered=catalog.OnlineTableSpecTriggeredSchedulingPolicy(),
    perform_full_copy=True,
)
ot = catalog.OnlineTable(name=online_name, spec=spec)
try:
    res = w.online_tables.create(table=ot)
    print("create submitted", flush=True)
    final = res.result(timeout=datetime.timedelta(seconds=900))
    print("online table:", final.name, "state", final.status.detailed_state if final.status else None, flush=True)
except Exception as e:
    print("create err:", str(e)[:500], flush=True)
    # report current state if it exists
    try:
        cur = w.online_tables.get(online_name)
        print("current state", cur.status.detailed_state if cur.status else None, flush=True)
    except Exception as e2:
        print("get err", str(e2)[:300], flush=True)
