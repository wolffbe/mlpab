import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import ml

w = WorkspaceClient()
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
STORE = f"{PREFIX}-store"

existing = {s.name: s for s in w.feature_store.list_online_stores()}
if STORE in existing:
    s = existing[STORE]
    print("exists:", s.name, s.state)
else:
    print("creating online store", STORE)
    s = w.feature_store.create_online_store(
        online_store=ml.OnlineStore(name=STORE, capacity="CU_1")
    )
    print("create returned state:", s.state)

# poll until AVAILABLE
deadline = time.time() + 1500
while True:
    s = w.feature_store.get_online_store(name=STORE)
    st = s.state.value if s.state else None
    print("state:", st, flush=True)
    if st == "AVAILABLE":
        break
    if st in ("STOPPED",) or time.time() > deadline:
        raise RuntimeError(f"store not available: {st}")
    time.sleep(15)

print("ONLINE STORE READY:", s.name, s.state, s.capacity)
