"""Publish predictions table to online store (now that PK is added)."""
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode

w = WorkspaceClient()
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
CATALOG = SCHEMA.split(".")[0]
DB = SCHEMA.split(".")[1]

PRED_NAME = "airqpredf4aae3"
ONLINE_STORE = f"{PREFIX}-pred-store"

source_table = f"{CATALOG}.{DB}.{PRED_NAME}"
online_table = f"{CATALOG}.{DB}.{PRED_NAME}"

print(f"Source table: {source_table}")
print(f"Online store: {ONLINE_STORE}")
print(f"Online table: {online_table}")

# First verify the online store exists
try:
    store = w.feature_store.get_online_store(name=ONLINE_STORE)
    print(f"Online store status: {store}")
except Exception as e:
    print(f"Online store not found, creating: {e}")
    from databricks.sdk.service.ml import OnlineStore
    try:
        store = w.feature_store.create_online_store(OnlineStore(name=ONLINE_STORE, capacity="CU_1"))
        print(f"Created online store: {store}")
    except Exception as e2:
        print(f"Create online store failed: {e2}")

# Publish the table
try:
    result = w.feature_store.publish_table(
        source_table_name=source_table,
        publish_spec=PublishSpec(
            online_store=ONLINE_STORE,
            online_table_name=online_table,
            publish_mode=PublishSpecPublishMode.TRIGGERED,
        ),
    )
    print(f"Publish result: {result}")
    print("SUCCESS: Table published to online store")
except Exception as e:
    print(f"publish_table failed: {e}")
    # Try SNAPSHOT mode
    try:
        result2 = w.feature_store.publish_table(
            source_table_name=source_table,
            publish_spec=PublishSpec(
                online_store=ONLINE_STORE,
                online_table_name=online_table,
                publish_mode=PublishSpecPublishMode.SNAPSHOT,
            ),
        )
        print(f"Publish (SNAPSHOT) result: {result2}")
        print("SUCCESS: Table published with SNAPSHOT mode")
    except Exception as e2:
        print(f"SNAPSHOT also failed: {e2}")

# List online tables in the store
try:
    stores = list(w.feature_store.list_online_stores())
    print(f"Online stores: {[s.name for s in stores]}")
except Exception as e:
    print(f"List stores: {e}")
