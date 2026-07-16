import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import (
    OnlineStore, OnlineStoreState, PublishSpec, PublishSpecPublishMode,
)

w = WorkspaceClient()
CATALOG, SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SRC = f"{CATALOG}.{SCHEMA}.scoredbfc4ef"
STORE = f"{PREFIX}-scorestore"
ONLINE_TBL = f"{CATALOG}.{SCHEMA}.scoredbfc4ef_online"

# 1. Create (or reuse) the online store -- a Lakebase-backed online feature store.
try:
    st = w.feature_store.get_online_store(STORE)
    print("online store exists:", st.state)
except Exception:
    print("creating online store", STORE)
    st = w.feature_store.create_online_store(OnlineStore(name=STORE, capacity="CU_1"))
    print("created, state:", st.state)

# wait for AVAILABLE
for _ in range(120):
    st = w.feature_store.get_online_store(STORE)
    if st.state == OnlineStoreState.AVAILABLE:
        break
    print("  waiting, state:", st.state)
    time.sleep(15)
print("online store state:", st.state)

# 2. Publish the feature table to the online store (creates a synced online table).
print("publishing", SRC, "->", ONLINE_TBL)
resp = w.feature_store.publish_table(
    source_table_name=SRC,
    publish_spec=PublishSpec(
        online_store=STORE,
        online_table_name=ONLINE_TBL,
        publish_mode=PublishSpecPublishMode.TRIGGERED,
    ),
)
print("publish response:", resp)
print("DONE online")
