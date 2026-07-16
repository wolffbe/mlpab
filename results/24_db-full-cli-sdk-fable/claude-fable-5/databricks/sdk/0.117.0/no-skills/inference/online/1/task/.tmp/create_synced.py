import os
import time

import databricks.sdk
import databricks.sdk.service.database as db

w = databricks.sdk.WorkspaceClient()
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]
catalog, schema_name = schema.split(".")
inst_name = f"{prefix}-online"
src = f"{schema}.profiles44b75e"
online_name = f"{schema}.profiles44b75e_online"

st = db.SyncedDatabaseTable(
    name=online_name,
    database_instance_name=inst_name,
    logical_database_name="databricks_postgres",
    spec=db.SyncedTableSpec(
        source_table_full_name=src,
        primary_key_columns=["account_id"],
        scheduling_policy=db.SyncedTableSchedulingPolicy.SNAPSHOT,
        create_database_objects_if_missing=True,
        new_pipeline_spec=db.NewPipelineSpec(
            storage_catalog=catalog, storage_schema=schema_name
        ),
    ),
)
try:
    res = w.database.create_synced_database_table(st)
    print("created:", res.name)
except Exception as e:
    print("create error:", e)

deadline = time.time() + 1500
while time.time() < deadline:
    cur = w.database.get_synced_database_table(online_name)
    s = cur.data_synchronization_status
    detailed = s.detailed_state.value if s and s.detailed_state else None
    msg = s.message if s else None
    print(time.strftime("%H:%M:%S"), detailed, "|", msg)
    if detailed in ("ONLINE", "ONLINE_TRIGGERED_UPDATE", "ONLINE_NO_PENDING_UPDATE",
                    "ONLINE_PIPELINE_FAILED"):
        break
    if detailed in ("FAILED", "OFFLINE_FAILED", "SYNCED_TABLE_FAILED"):
        raise SystemExit(f"sync failed: {msg}")
    time.sleep(20)
print("final:", detailed)
