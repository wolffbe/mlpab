import os, time
import databricks.sdk
from databricks.sdk.service import database as db

w = databricks.sdk.WorkspaceClient()
CAT, SCH = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]

inst_name = f"{PREFIX}-recs"
src = f"{CAT}.{SCH}.recs2ead15"
synced_name = f"{CAT}.{SCH}.recs2ead15_synced"

# 1. Create Lakebase database instance (prefixed)
existing = None
try:
    existing = w.database.get_database_instance(name=inst_name)
    print("instance already exists:", inst_name, existing.state)
except Exception:
    pass

if existing is None:
    print("creating instance", inst_name)
    inst = db.DatabaseInstance(name=inst_name, capacity="CU_1")
    existing = w.database.create_database_instance_and_wait(database_instance=inst)
    print("instance state:", existing.state)

# 2. Create synced table from the Delta feature table
spec = db.SyncedTableSpec(
    source_table_full_name=src,
    primary_key_columns=["rec_id"],
    scheduling_policy=db.SyncedTableSchedulingPolicy.SNAPSHOT,
    create_database_objects_if_missing=True,
    new_pipeline_spec=db.NewPipelineSpec(storage_catalog=CAT, storage_schema=SCH),
)
st = db.SyncedDatabaseTable(
    name=synced_name,
    database_instance_name=inst_name,
    logical_database_name="databricks_postgres",
    spec=spec,
)
print("creating synced table", synced_name)
res = w.database.create_synced_database_table(synced_table=st)
print("created:", res.name)
print("prov state:", res.unity_catalog_provisioning_state)
