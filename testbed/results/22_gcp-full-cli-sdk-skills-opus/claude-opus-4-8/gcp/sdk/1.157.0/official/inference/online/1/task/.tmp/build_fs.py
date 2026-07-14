import os, datetime
import google.cloud.aiplatform as aiplatform

project = os.environ['GCP_PROJECT']
loc = os.environ['GCP_LOCATION']
prefix = os.environ['MLPAB_GCP_PREFIX']
dataset = os.environ['GCP_BQ_DATASET']
aiplatform.init(project=project, location=loc, api_transport='rest')

fs_id = f'{prefix}_profilesaf22bf_fs'
et_id = 'profilesaf22bf'

try:
    fs = aiplatform.Featurestore(fs_id)
    print('featurestore exists', fs.resource_name, flush=True)
except Exception:
    fs = aiplatform.Featurestore.create(
        featurestore_id=fs_id, online_store_fixed_node_count=1, labels={'run': prefix})
    print('created featurestore', fs.resource_name, flush=True)

try:
    et = fs.get_entity_type(et_id)
    print('entity type exists', flush=True)
except Exception:
    et = fs.create_entity_type(entity_type_id=et_id,
                               description='account profiles, key=account_id')
    print('created entity type', et.resource_name, flush=True)

existing = {f.name for f in et.list_features()}
cfg = {fid: {'value_type': 'DOUBLE'} for fid in ['f1', 'f2', 'f3', 'f4'] if fid not in existing}
if cfg:
    et.batch_create_features(feature_configs=cfg)
    print('created features', list(cfg), flush=True)
else:
    print('features exist', flush=True)

bq_uri = f'bq://{project}.{dataset}.profilesaf22bf'
ft = datetime.datetime(2024, 1, 1, 0, 0, 0)
et.ingest_from_bq(feature_ids=['f1', 'f2', 'f3', 'f4'], feature_time=ft,
                  bq_source_uri=bq_uri, entity_id_field='account_id')
print('INGEST DONE', flush=True)
