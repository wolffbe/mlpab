"""Try Feature Store publish_table with different online_table_name."""
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode
from databricks.sdk.service.database import (
    SyncedDatabaseTable, SyncedTableSpec, SyncedTableSchedulingPolicy
)

w = WorkspaceClient()
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
CATALOG = SCHEMA.split(".")[0]
DB = SCHEMA.split(".")[1]

PRED_NAME = "airqpredf4aae3"
ONLINE_STORE = f"{PREFIX}-pred-store"
source_table = f"{CATALOG}.{DB}.{PRED_NAME}"

# Try Feature Store publish with different online table names
print("=== Feature Store Publish Attempts ===")
for online_name in [
    f"{CATALOG}.{DB}.{PRED_NAME}ot",
    f"{CATALOG}.{DB}.{PRED_NAME}_online",
    f"{CATALOG}.{DB}.{PRED_NAME}_pub",
]:
    try:
        result = w.feature_store.publish_table(
            source_table_name=source_table,
            publish_spec=PublishSpec(
                online_store=ONLINE_STORE,
                online_table_name=online_name,
                publish_mode=PublishSpecPublishMode.TRIGGERED,
            ),
        )
        print(f"SUCCESS online_table_name={online_name}: {result}")
        break
    except Exception as e:
        print(f"FAIL {online_name}: {e}")

print()
print("=== Synced Table Approach ===")
# Try synced table with different names
DB_INSTANCE_NAME = f"{PREFIX}-lakebase"
for synced_name in [
    f"{CATALOG}.{DB}.{PRED_NAME}ot",
    f"{CATALOG}.{DB}.{PRED_NAME}_online",
]:
    try:
        result = w.database.create_synced_database_table(
            SyncedDatabaseTable(
                name=synced_name,
                database_instance_name=DB_INSTANCE_NAME,
                logical_database_name=DB,
                spec=SyncedTableSpec(
                    source_table_full_name=source_table,
                    primary_key_columns=["date"],
                    scheduling_policy=SyncedTableSchedulingPolicy.TRIGGERED,
                    create_database_objects_if_missing=True,
                ),
            )
        )
        print(f"SUCCESS synced_name={synced_name}: {result.name} UC={result.unity_catalog_provisioning_state}")
        break
    except Exception as e:
        print(f"FAIL synced {synced_name}: {e}")
