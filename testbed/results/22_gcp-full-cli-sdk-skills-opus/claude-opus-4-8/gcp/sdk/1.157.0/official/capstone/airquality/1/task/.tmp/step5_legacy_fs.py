import os
from google.cloud import aiplatform_v1 as a
from google.protobuf.timestamp_pb2 import Timestamp

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']; ds = os.environ['GCP_BQ_DATASET']
prefix = os.environ['MLPAB_GCP_PREFIX']
ep = f"{loc}-aiplatform.googleapis.com"; copts = {"api_endpoint": ep}
parent = f"projects/{proj}/locations/{loc}"
svc = a.FeaturestoreServiceClient(transport="rest", client_options=copts)

fs_id = f"{prefix}_airqpred_fs"          # Vertex resource, prefixed
et_id = "airqpredf3f1d8"                 # entity type = predictions feature table
fs_path = f"{parent}/featurestores/{fs_id}"
et_path = f"{fs_path}/entityTypes/{et_id}"

# 1) Featurestore with ONLINE serving enabled (low-latency Bigtable) -> distinguishes online/offline
try:
    op = svc.create_featurestore(
        parent=parent, featurestore_id=fs_id,
        featurestore=a.Featurestore(
            online_serving_config=a.Featurestore.OnlineServingConfig(fixed_node_count=1)),
    )
    fsr = op.result(timeout=1200)
    print("featurestore created:", fsr.name)
except Exception as e:
    print("featurestore:", type(e).__name__, str(e)[:200])

# 2) EntityType (entities keyed by `date`)
try:
    op = svc.create_entity_type(
        parent=fs_path, entity_type_id=et_id,
        entity_type=a.EntityType(description="PM2.5 forecast predictions keyed by date"))
    print("entity type:", op.result(timeout=600).name)
except Exception as e:
    print("entity type:", type(e).__name__, str(e)[:200])

# 3) Feature pm25_pred (DOUBLE)
try:
    op = svc.batch_create_features(
        parent=et_path,
        requests=[a.CreateFeatureRequest(
            feature=a.Feature(value_type=a.Feature.ValueType.DOUBLE,
                              description="predicted PM2.5"),
            feature_id="pm25_pred")])
    print("feature:", [f.name for f in op.result(timeout=600).features])
except Exception as e:
    print("feature:", type(e).__name__, str(e)[:200])

# 4) Ingest predictions from BigQuery -> online + offline store
ft = Timestamp(); ft.FromSeconds(1750000000)   # constant feature time (~2025-06)
try:
    op = svc.import_feature_values(
        request=a.ImportFeatureValuesRequest(
            entity_type=et_path,
            bigquery_source=a.BigQuerySource(
                input_uri=f"bq://{proj}.{ds}.airqpredf3f1d8"),
            entity_id_field="date",
            feature_specs=[a.ImportFeatureValuesRequest.FeatureSpec(id="pm25_pred")],
            feature_time=ft,
            worker_count=1))
    res = op.result(timeout=1200)
    print("ingest: imported=%d invalid=%d" % (res.imported_feature_value_count, res.invalid_row_count))
except Exception as e:
    print("ingest:", type(e).__name__, str(e)[:300])
