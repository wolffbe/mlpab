import databricks.sdk as s
import databricks.sdk.service.database as d
import os, time

w = s.WorkspaceClient()
catalog, schema = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
inst_name = f"{prefix}-lakebase"
src = f"{catalog}.{schema}.scores3380ed"
synced_name = f"{catalog}.{schema}.scores3380ed_online"

spec = d.SyncedTableSpec(
    source_table_full_name=src,
    primary_key_columns=["account_id"],
    scheduling_policy=d.SyncedTableSchedulingPolicy.SNAPSHOT,
    create_database_objects_if_missing=True,
)
st = d.SyncedDatabaseTable(
    name=synced_name,
    database_instance_name=inst_name,
    logical_database_name="databricks_postgres",
    spec=spec,
)
try:
    w.database.delete_synced_database_table(synced_name)
    time.sleep(5)
except Exception as e:
    print("pre-delete:", repr(e)[:120])

res = w.database.create_synced_database_table(st)
print("created synced table:", res.name)

for i in range(90):
    cur = w.database.get_synced_database_table(synced_name)
    dss = cur.data_synchronization_status
    state = None
    if dss and dss.detailed_state:
        state = dss.detailed_state.value
    prov = cur.unity_catalog_provisioning_state.value if cur.unity_catalog_provisioning_state else None
    print(i, "prov=", prov, "sync=", state)
    if state in ("SYNCED_NO_PENDING_UPDATE", "ONLINE_NO_PENDING_UPDATE", "ONLINE"):
        break
    if state and "FAIL" in state:
        print("FAILED detail:", dss)
        break
    if prov == "FAILED":
        print("PROV FAILED")
        break
    time.sleep(10)
print("FINAL synced state:", state, "prov:", prov, "name:", synced_name)
