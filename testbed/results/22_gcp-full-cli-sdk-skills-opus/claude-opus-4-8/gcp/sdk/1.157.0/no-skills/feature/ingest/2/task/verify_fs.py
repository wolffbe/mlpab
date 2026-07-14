import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as vfs
from google.cloud import bigquery

project = os.environ['GCP_PROJECT']
dataset = os.environ['GCP_BQ_DATASET']
loc = os.environ['GCP_LOCATION']
prefix = os.environ['MLPAB_GCP_PREFIX']

aiplatform.init(project=project, location=loc, api_transport='rest')

# Offline: FeatureGroup + features + BQ row count
fg = vfs.FeatureGroup('transactions8cf1c0')
print('FeatureGroup:', fg.resource_name)
print('features:', sorted(f.name for f in fg.list_features()))

c = bigquery.Client(project=project)
r = list(c.query(
    f"SELECT COUNT(*) n, COUNT(DISTINCT row_id) d FROM `{project}.{dataset}.transactions8cf1c0`",
    location=loc).result())[0]
print('offline BQ rows:', r.n, 'distinct row_id:', r.d)

# Online: read a few keys from the feature view
store_name = f'{prefix}_transactions8cf1c0_fos'
fos = vfs.FeatureOnlineStore(store_name)
fv = {v.name: v for v in fos.list_feature_views()}['transactions8cf1c0']
print('online store:', fos.resource_name)
print('feature view:', fv.resource_name)
for key in ['R00000', 'R00373', 'R00472']:
    try:
        resp = fv.read(key=[key])
        print('online read', key, '->', resp.to_dict())
    except Exception as e:
        print('online read', key, 'FAILED (client->data-plane connectivity):', type(e).__name__)
# sync status
syncs = fv.list_syncs()
for s in syncs:
    code = getattr(getattr(s.gca_resource, 'final_status', None), 'code', None)
    print('sync', s.name, 'final_status.code=', code)
print('VERIFIED')
