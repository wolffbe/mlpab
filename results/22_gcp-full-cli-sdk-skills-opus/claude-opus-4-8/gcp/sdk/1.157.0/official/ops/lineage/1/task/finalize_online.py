import os, time
import vertexai
from vertexai.resources.preview import feature_store as vfs
from vertexai.resources.preview.feature_store import utils as fsutils

proj=os.environ['GCP_PROJECT']; ds=os.environ['GCP_BQ_DATASET']; loc=os.environ['GCP_LOCATION']
prefix=os.environ['MLPAB_GCP_PREFIX']
vertexai.init(project=proj, location=loc, api_transport="rest")
store_name=f"{prefix}_online_store"

store=None
for i in range(30):
    try:
        store=vfs.FeatureOnlineStore(store_name)
        state=getattr(store.gca_resource,'state',None)
        print(f"store state poll {i}: {state}")
        if str(state)=="State.STABLE" or (state is not None and int(state)==1):
            break
    except Exception as e:
        print("  waiting store:", repr(e)[:100])
    time.sleep(15)

fv_name="derived55c41b"
src=fsutils.FeatureViewBigQuerySource(uri=f"bq://{proj}.{ds}.derived55c41b", entity_id_columns=["row_id"])
fv=None
try:
    for v in store.list_feature_views():
        if v.name==fv_name: fv=v; print("fv exists"); break
except Exception as e:
    print("list fv err", repr(e)[:120])
if fv is None:
    fv=store.create_feature_view(name=fv_name, source=src, labels={"version":"1"})
    print("created feature view", fv_name)

try:
    fv.sync(); print("sync triggered")
    for i in range(12):
        syncs=fv.list_syncs()
        print(f"  sync poll {i}: {len(syncs)} sync(s)")
        done=False
        for s in syncs:
            fs=getattr(s.gca_resource,'final_status',None)
            if fs is not None and getattr(fs,'code',1)==0: done=True
        if done: print("SYNC DONE"); break
        time.sleep(20)
except Exception as e:
    print("sync err", repr(e)[:200])

print("ONLINE STORE:", store.resource_name)
print("FEATURE VIEW:", fv.resource_name)
