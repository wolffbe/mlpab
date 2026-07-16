import databricks.sdk as s
import databricks.sdk.service.catalog as c
import os, time

w = s.WorkspaceClient()
catalog, schema = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
src = f"{catalog}.{schema}.scores3380ed"
ot_name = f"{catalog}.{schema}.scores3380ed_online"

# enable change data feed on source (needed by online tables)
wh = "4dfab06c923fe3cc"
w.statement_execution.execute_statement(
    warehouse_id=wh,
    statement=f"ALTER TABLE {src} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)",
    wait_timeout="50s")
print("CDF enabled")

spec = c.OnlineTableSpec(
    source_table_full_name=src,
    primary_key_columns=["account_id"],
    perform_full_copy=True,
    run_triggered=c.OnlineTableSpecTriggeredSchedulingPolicy(),
)
try:
    w.online_tables.delete(name=ot_name)
    time.sleep(5)
except Exception as e:
    print("delete pre:", repr(e)[:120])

ot = c.OnlineTable(name=ot_name, spec=spec)
res = w.online_tables.create(table=ot)
print("create submitted:", type(res).__name__)

# poll for readiness
for i in range(60):
    cur = w.online_tables.get(name=ot_name)
    state = cur.status.detailed_state.value if (cur.status and cur.status.detailed_state) else "UNKNOWN"
    print(i, state)
    if state in ("ONLINE_NO_PENDING_UPDATE", "ONLINE", "ONLINE_TRIGGERED_UPDATE"):
        break
    if "FAILED" in state or "OFFLINE" in state:
        print("MESSAGE:", cur.status.message if cur.status else None)
        break
    time.sleep(10)
print("final online table state:", state, "name:", ot_name)
