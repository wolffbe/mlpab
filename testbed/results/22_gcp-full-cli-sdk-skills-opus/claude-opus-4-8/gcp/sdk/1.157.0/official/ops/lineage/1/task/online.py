import os, time
import vertexai
from vertexai.resources.preview import feature_store as vfs
from vertexai.resources.preview.feature_store import utils as fsutils

proj=os.environ['GCP_PROJECT']; ds=os.environ['GCP_BQ_DATASET']; loc=os.environ['GCP_LOCATION']
prefix=os.environ['MLPAB_GCP_PREFIX']
vertexai.init(project=proj, location=loc, api_transport="rest")

store_name = f"{prefix}_online_store"
# 1) Optimized online store = low-latency / real-time serving path
store = None
try:
    for s in vfs.FeatureOnlineStore.list():
        if s.name == store_name:
            store = s; print("online store exists", store_name); break
except Exception as e:
    print("list err", repr(e)[:120])
if store is None:
    store = vfs.FeatureOnlineStore.create_optimized_store(name=store_name)
    print("created online store", store_name)

# 2) FeatureView over the derived feature data, keyed by row_id
fv_name = "derived55c41b"
src = fsutils.FeatureViewBigQuerySource(
    uri=f"bq://{proj}.{ds}.derived55c41b", entity_id_columns=["row_id"])
fv = None
try:
    for v in store.list_feature_views():
        if v.name == fv_name:
            fv = v; print("feature view exists", fv_name); break
except Exception as e:
    print("list fv err", repr(e)[:120])
if fv is None:
    fv = store.create_feature_view(name=fv_name, source=src, labels={"version": "1"})
    print("created feature view", fv_name)

# 3) Sync offline -> online so lookups return data
try:
    sync = fv.sync()
    print("sync started")
    for i in range(20):
        syncs = fv.list_syncs()
        states = [str(getattr(s.gca_resource.run_time, 'end_time', '')) for s in syncs]
        done = any(getattr(s.gca_resource, 'final_status', None) and s.gca_resource.final_status.code == 0 for s in syncs)
        print(f"  sync poll {i}: {len(syncs)} sync(s)")
        if done:
            print("sync completed"); break
        time.sleep(20)
except Exception as e:
    print("sync err", repr(e)[:200])

print("ONLINE STORE:", store.resource_name)
print("FEATURE VIEW:", fv.resource_name)
