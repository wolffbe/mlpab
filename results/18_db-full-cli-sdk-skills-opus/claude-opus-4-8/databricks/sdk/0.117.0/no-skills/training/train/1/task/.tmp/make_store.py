import time
import databricks.sdk as dsdk
import databricks.sdk.service.ml as ml

w = dsdk.WorkspaceClient()
STORE = "mlpab2138eb-predstore"   # prefixed Lakebase/online store instance (DNS-compliant)

existing = None
for s in w.feature_store.list_online_stores():
    if s.name == STORE:
        existing = s
print("existing:", existing.state if existing else None)

if existing is None:
    for cap in ["CU_1", "CU_2", "CU_4"]:
        try:
            os_obj = w.feature_store.create_online_store(
                ml.OnlineStore(name=STORE, capacity=cap))
            print("created online store with capacity", cap, "state:", os_obj.state)
            existing = os_obj
            break
        except Exception as e:
            print(f"capacity {cap} failed:", repr(e)[:250])
    if existing is None:
        raise SystemExit("could not create online store")

# poll until AVAILABLE
deadline = time.monotonic() + 1500
while True:
    s = w.feature_store.get_online_store(STORE)
    print("state:", s.state)
    if s.state == ml.OnlineStoreState.AVAILABLE:
        break
    if s.state in (ml.OnlineStoreState.STOPPED,):
        raise SystemExit(f"store in bad state {s.state}")
    if time.monotonic() > deadline:
        raise SystemExit("timeout waiting for online store")
    time.sleep(15)
print("ONLINE STORE AVAILABLE:", STORE)
