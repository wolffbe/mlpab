import os, time
import google.cloud.aiplatform as aiplatform
import vertexai
from vertexai.resources.preview import feature_store as fs
from google.api_core.exceptions import ResourceExhausted, NotFound

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']; ds = os.environ['GCP_BQ_DATASET']
aiplatform.init(project=proj, location=loc, api_transport="rest")
vertexai.init(project=proj, location=loc)

bq_uri = f"bq://{proj}.{ds}.accountsed4daa"
STORE = "accountsed4daa_online"
VIEW = "accountsed4daa"

# 1) Online store (optimized -> low-latency online serving). Retry through the
#    shared project-wide FeatureOnlineStores quota contention from concurrent runs.
store = None
try:
    store = fs.FeatureOnlineStore(STORE)
    print("online store exists:", store.resource_name)
except NotFound:
    pass

attempts = 8
for i in range(attempts):
    if store is not None:
        break
    try:
        store = fs.FeatureOnlineStore.create_optimized_store(name=STORE)
        print("online store created:", store.resource_name)
    except ResourceExhausted as e:
        print(f"attempt {i+1}/{attempts}: quota exhausted, waiting...", str(e)[:80])
        # clean up any half-created store so retries can reuse the id
        try:
            fs.FeatureOnlineStore(STORE).delete(force=True)
        except Exception:
            pass
        if i < attempts - 1:
            time.sleep(45)

if store is None:
    print("ONLINE_BLOCKED: FeatureOnlineStores quota exhausted after retries")
    raise SystemExit(2)

# 2) FeatureView from the deduped BQ table, keyed by row_id
views = {v.name: v for v in store.list_feature_views()}
if VIEW in views:
    fv = views[VIEW]
    print("feature view exists:", fv.resource_name)
else:
    fv = store.create_feature_view(
        name=VIEW,
        source=fs.FeatureViewBigQuerySource(uri=bq_uri, entity_id_columns=["row_id"]),
    )
    print("feature view created:", fv.resource_name)

# 3) Sync BQ -> online store (blocking until sync completes)
sync_resp = fv.sync()
print("sync started:", sync_resp)
sync_name = sync_resp.resource_name if hasattr(sync_resp, "resource_name") else str(sync_resp)
# poll sync to completion
for _ in range(60):
    s = fv.get_sync(sync_name.split("/")[-1]) if "/" in sync_name else None
    try:
        done = s.gca_resource.run_time.end_time.seconds > 0 if s else False
    except Exception:
        done = False
    if done:
        print("sync complete")
        break
    time.sleep(20)
print("DONE_ONLINE")
