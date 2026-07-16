import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as vfs

project = os.environ['GCP_PROJECT']
dataset = os.environ['GCP_BQ_DATASET']
loc = os.environ['GCP_LOCATION']
prefix = os.environ['MLPAB_GCP_PREFIX']

aiplatform.init(project=project, location=loc, api_transport='rest')

bq_uri = f'bq://{project}.{dataset}.transactions8cf1c0'

# 1. FeatureGroup (offline, BigQuery-backed) named as the task requires
try:
    fg = vfs.FeatureGroup('transactions8cf1c0')
    print('FeatureGroup already exists:', fg.resource_name)
except Exception:
    fg = vfs.FeatureGroup.create(
        'transactions8cf1c0',
        source=vfs.FeatureGroupBigQuerySource(uri=bq_uri, entity_id_columns=['row_id']),
        description='transactions feature table v1; record key row_id, event-time event_time (epoch ms)',
        labels={'version': '1'},
    )
    print('Created FeatureGroup:', fg.resource_name)

# 2. Register features (non-key, non-timestamp columns)
existing = {f.name for f in fg.list_features()}
for feat in ['account_id', 'event_time', 'amount', 'category']:
    if feat in existing:
        print('feature exists:', feat)
        continue
    fg.create_feature(feat, description=f'feature column {feat} from transactions8cf1c0')
    print('created feature:', feat)

print('features now:', [f.name for f in fg.list_features()])
