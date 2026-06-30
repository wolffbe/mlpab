import datetime
import databricks.sdk as dsdk
import databricks.sdk.service.catalog as cat

w = dsdk.WorkspaceClient()
CATALOG, SCHEMA = "workspace", "mlpab2138eb"
SRC = f"{CATALOG}.{SCHEMA}.predictionsa834e5"
ONLINE = f"{CATALOG}.{SCHEMA}.predictionsa834e5_online"

# delete if exists
try:
    w.online_tables.delete(name=ONLINE)
    print("deleted existing online table")
except Exception as e:
    print("no existing:", repr(e)[:120])

spec = cat.OnlineTableSpec(
    source_table_full_name=SRC,
    primary_key_columns=["row_id"],
    perform_full_copy=True,
    run_triggered=cat.OnlineTableSpecTriggeredSchedulingPolicy(),
)
ot = cat.OnlineTable(name=ONLINE, spec=spec)
res = w.online_tables.create_and_wait(table=ot,
        timeout=datetime.timedelta(minutes=20))
print("online table:", res.name)
print("state:", res.unity_catalog_provisioning_state)
if res.status:
    print("detailed:", res.status.detailed_state)
print("serving_url:", res.table_serving_url)
