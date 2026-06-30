import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import ml

w = WorkspaceClient()
STORE = "mlpab4bb10d-fs74f1ef"  # prefixed name (database instance / endpoint), DNS-compliant

store = ml.OnlineStore(name=STORE, capacity="CU_1")
res = w.feature_store.create_online_store(online_store=store)
print("create returned:", res.name, res.state)

# poll until available
for _ in range(120):
    s = w.feature_store.get_online_store(name=STORE)
    print("state:", s.state)
    if s.state == ml.OnlineStoreState.AVAILABLE:
        break
    if s.state in (ml.OnlineStoreState.STOPPED,):
        break
    time.sleep(15)
print("final:", w.feature_store.get_online_store(name=STORE).state)
