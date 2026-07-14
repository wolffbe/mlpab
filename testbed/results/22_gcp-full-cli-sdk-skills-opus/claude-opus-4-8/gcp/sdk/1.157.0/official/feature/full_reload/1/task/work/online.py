import os
import time
for v in ("GRPC_PROXY", "grpc_proxy"):
    os.environ.pop(v, None)
import vertexai
from vertexai.resources.preview import feature_store as fs

project = os.environ['GCP_PROJECT']
location = os.environ['GCP_LOCATION']
prefix = os.environ['MLPAB_GCP_PREFIX']

vertexai.init(project=project, location=location, api_transport="rest")

store_name = f"{prefix}_customerscd1186_online"
view_name = f"{prefix}_customerscd1186_2"
fg2 = f"{prefix}_customerscd1186_2"
features = [f"{fg2}.{c}" for c in ("full_name", "balance", "currency", "updated_at")]

# 1. Optimized online store (low-latency serving)
from google.api_core.exceptions import ResourceExhausted
try:
    store = fs.FeatureOnlineStore(store_name)
    print("online store exists:", store.name)
except Exception:
    store = None
    for attempt in range(15):
        try:
            print(f"creating optimized online store (attempt {attempt+1})...")
            store = fs.FeatureOnlineStore.create_optimized_store(name=store_name)
            print("created online store:", store.name, store.feature_online_store_type)
            break
        except ResourceExhausted as e:
            print("quota exhausted, waiting 60s:", str(e)[:120])
            time.sleep(60)
    if store is None:
        raise SystemExit("QUOTA: FeatureOnlineStores quota remained exhausted")

# 2. Feature view over v2 registry features
existing_views = {v.name for v in store.list_feature_views()}
if view_name in existing_views:
    fv = fs.FeatureView(view_name, feature_online_store_id=store_name)
    print("feature view exists:", fv.name)
else:
    src = fs.FeatureViewRegistrySource(features=features)
    fv = store.create_feature_view(name=view_name, source=src)
    print("created feature view:", fv.name)

# 3. Trigger a data sync (offline v2 -> online store)
sync = fv.sync()
print("started sync:", sync.name)

# 4. Wait for sync completion
deadline = time.time() + 1500
last = None
while time.time() < deadline:
    s = fv.get_sync(sync.name.split("/")[-1])
    r = s.gca_resource
    final = getattr(r.run_time, "end_time", None)
    status = getattr(getattr(r, "final_status", None), "code", None)
    state = f"end={final} status_code={status}"
    if state != last:
        print("sync:", state)
        last = state
    if final and final.seconds:
        print("SYNC COMPLETE")
        break
    time.sleep(20)
else:
    print("sync still running at deadline")
