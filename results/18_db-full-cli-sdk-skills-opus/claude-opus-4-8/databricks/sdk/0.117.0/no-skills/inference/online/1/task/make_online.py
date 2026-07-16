from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

w = WorkspaceClient()
SCHEMA = "workspace.mlpab6cf45f"
SRC = f"{SCHEMA}.profilesf45007"
ONLINE = f"{SCHEMA}.profilesf45007_online"

spec = catalog.OnlineTableSpec(
    source_table_full_name=SRC,
    primary_key_columns=["account_id"],
    perform_full_copy=True,
    run_triggered=catalog.OnlineTableSpecTriggeredSchedulingPolicy(),
)
ot = catalog.OnlineTable(name=ONLINE, spec=spec)

# delete any pre-existing
try:
    w.online_tables.delete(name=ONLINE)
    print("deleted existing online table")
except Exception as e:
    print("no existing online table:", str(e)[:120])

print("creating online table...")
waiter = w.online_tables.create(table=ot)
res = waiter.result(timeout=__import__("datetime").timedelta(minutes=20))
print("online table created")
print("name:", res.name)
print("serving_url:", res.table_serving_url)
print("status:", res.status)
print("provisioning_state:", res.unity_catalog_provisioning_state)
