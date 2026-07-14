from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import OnlineStore, PublishSpec, PublishSpecPublishMode
import os, time

w = WorkspaceClient()
cat, sch = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
SRC = "%s.%s.customers302d18" % (cat, sch)
STORE = "%s-custonline" % prefix
ONLINE_TBL = "%s.%s.customers302d18_online" % (cat, sch)

# 1) create (or reuse) online store
try:
    st = w.feature_store.get_online_store(STORE)
    print("online store exists:", st.name, st.state)
except Exception:
    print("creating online store", STORE)
    st = w.feature_store.create_online_store(OnlineStore(name=STORE, capacity="CU_1"))
    print("created:", st.name, st.state)

# wait until AVAILABLE
for _ in range(120):
    st = w.feature_store.get_online_store(STORE)
    s = st.state.value if st.state else None
    if s == "AVAILABLE":
        break
    print("  store state:", s)
    time.sleep(15)
print("online store final state:", st.state)

# 2) publish v2 table to the online store
print("publishing", SRC, "->", ONLINE_TBL)
resp = w.feature_store.publish_table(
    source_table_name=SRC,
    publish_spec=PublishSpec(
        online_store=STORE,
        online_table_name=ONLINE_TBL,
        publish_mode=PublishSpecPublishMode.SNAPSHOT,
    ),
)
print("publish response:", resp)
