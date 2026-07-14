from google.cloud import aiplatform_v1 as v1
from google.protobuf import timestamp_pb2
import os, time
proj=os.environ['GCP_PROJECT']; loc=os.environ['GCP_LOCATION']; ds=os.environ['GCP_BQ_DATASET']; pref=os.environ['MLPAB_GCP_PREFIX']
ep=f"{loc}-aiplatform.googleapis.com"
c=v1.FeaturestoreServiceClient(client_options={"api_endpoint":ep}, transport="rest")
FS_ID=f"{pref}_ccfs"; ET_ID="transaction"
fs_path=c.featurestore_path(proj,loc,FS_ID)

# EntityType keyed by transaction_id
try:
    op=c.create_entity_type(parent=fs_path, entity_type_id=ET_ID,
        entity_type=v1.EntityType(description="scored transactions"))
    et=op.result(timeout=300); print("entity type:", et.name)
except Exception as e:
    print("ET note:", type(e).__name__, str(e)[:150])
et_path=c.entity_type_path(proj,loc,FS_ID,ET_ID)

# Feature fraud_probability (DOUBLE)
try:
    op=c.create_feature(parent=et_path, feature_id="fraud_probability",
        feature=v1.Feature(value_type=v1.Feature.ValueType.DOUBLE, description="P(fraud) in [0,1]"))
    ft=op.result(timeout=300); print("feature:", ft.name)
except Exception as e:
    print("Feat note:", type(e).__name__, str(e)[:150])

# Import from BQ predictions table -> online store
ts=timestamp_pb2.Timestamp(); ts.FromSeconds(int(time.time()))
req=v1.ImportFeatureValuesRequest(
    entity_type=et_path,
    bigquery_source=v1.BigQuerySource(input_uri=f"bq://{proj}.{ds}.ccpred76ccb2"),
    entity_id_field="transaction_id",
    feature_specs=[v1.ImportFeatureValuesRequest.FeatureSpec(id="fraud_probability")],
    feature_time=ts,
    worker_count=1)
op=c.import_feature_values(request=req)
print("importing feature values...")
r=op.result(timeout=1200)
print("imported rows:", r.imported_entity_count, "feature values:", r.imported_feature_value_count, "invalid:", r.invalid_row_count)
print("done")
