import os
import time
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as vfs

project = os.environ['GCP_PROJECT']
dataset = os.environ['GCP_BQ_DATASET']
loc = os.environ['GCP_LOCATION']
prefix = os.environ['MLPAB_GCP_PREFIX']

aiplatform.init(project=project, location=loc, api_transport='rest')

bq_uri = f'bq://{project}.{dataset}.transactions8cf1c0'
store_name = f'{prefix}_transactions8cf1c0_fos'
view_name = 'transactions8cf1c0'

# 1. Online store (optimized / serverless) for low-latency lookup
existing_stores = {s.name for s in vfs.FeatureOnlineStore.list()}
if store_name in existing_stores:
    fos = vfs.FeatureOnlineStore(store_name)
    print('online store exists:', fos.resource_name)
else:
    fos = vfs.FeatureOnlineStore.create_optimized_store(store_name)
    print('created online store:', fos.resource_name)

# 2. Feature view backed by the same BigQuery table, keyed by row_id
existing_views = {v.name: v for v in fos.list_feature_views()}
if view_name in existing_views:
    fv = existing_views[view_name]
    print('feature view exists:', fv.resource_name)
else:
    fv = fos.create_feature_view(
        view_name,
        source=vfs.FeatureViewBigQuerySource(uri=bq_uri, entity_id_columns=['row_id']),
    )
    print('created feature view:', fv.resource_name)

# 3. Trigger an on-demand sync to populate the online store
sync_resp = fv.sync()
print('sync triggered:', sync_resp)

# 4. Poll sync completion
deadline = 1800
start = 0
while start < deadline:
    syncs = fv.list_syncs()
    done = False
    for s in syncs:
        st = s.gca_resource
        end_time = getattr(getattr(st, 'run_time', None), 'end_time', None)
        final = getattr(getattr(st, 'final_status', None), 'code', None)
        if end_time and end_time.seconds:
            print('sync finished. final_status.code=', final)
            done = True
            break
    if done:
        break
    time.sleep(20)
    start += 20
    print('waiting for sync...', start, 's')

print('DONE online serving setup')
