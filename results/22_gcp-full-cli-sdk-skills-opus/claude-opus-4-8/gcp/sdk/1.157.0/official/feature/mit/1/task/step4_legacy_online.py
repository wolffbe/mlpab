import os, time
import google.cloud.aiplatform_v1 as v1
from google.protobuf.timestamp_pb2 import Timestamp
from google.api_core.exceptions import AlreadyExists, ResourceExhausted

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
API = f"{LOCATION}-aiplatform.googleapis.com"
parent = f"projects/{PROJECT}/locations/{LOCATION}"
BQ_URI = f"bq://{PROJECT}.{DATASET}.features347afc"

FS_ID = f"{PREFIX}_fs347afc"          # legacy featurestore (online serving) - prefixed Vertex resource
ET_ID = "features347afc"              # entity type = the feature table name, key = row_id
FEATURES = [
    ("account_id", v1.Feature.ValueType.STRING),
    ("event_time", v1.Feature.ValueType.INT64),
    ("amount_usd", v1.Feature.ValueType.DOUBLE),
    ("is_weekend", v1.Feature.ValueType.INT64),
    ("amount_7d",  v1.Feature.ValueType.DOUBLE),
]

fs_client = v1.FeaturestoreServiceClient(client_options={"api_endpoint": API}, transport="rest")
online = v1.FeaturestoreOnlineServingServiceClient(client_options={"api_endpoint": API}, transport="rest")

fs_path = fs_client.featurestore_path(PROJECT, LOCATION, FS_ID)
et_path = fs_client.entity_type_path(PROJECT, LOCATION, FS_ID, ET_ID)

# 1) featurestore with online serving (separate quota from FeatureOnlineStores)
def fs_exists():
    try:
        fs_client.get_featurestore(name=fs_path); return True
    except Exception:
        return False

if not fs_exists():
    try:
        fs = v1.Featurestore(online_serving_config=v1.Featurestore.OnlineServingConfig(fixed_node_count=1))
        fs_client.create_featurestore(
            request=v1.CreateFeaturestoreRequest(parent=parent, featurestore=fs, featurestore_id=FS_ID)
        ).result(timeout=1800)
        print("created featurestore", FS_ID)
    except AlreadyExists:
        print("featurestore exists")
    except ResourceExhausted as e:
        print("LEGACY_FEATURESTORE_QUOTA_EXHAUSTED:", str(e)[:200]); raise SystemExit(0)
else:
    print("featurestore exists")

# 2) entity type (keyed by row_id)
try:
    fs_client.create_entity_type(
        request=v1.CreateEntityTypeRequest(
            parent=fs_path, entity_type_id=ET_ID,
            entity_type=v1.EntityType(description="Derived transaction features v1; key=row_id"),
        )
    ).result(timeout=600)
    print("created entity type", ET_ID)
except AlreadyExists:
    print("entity type exists")

# 3) features
for fid, vt in FEATURES:
    try:
        fs_client.create_feature(
            request=v1.CreateFeatureRequest(parent=et_path, feature_id=fid, feature=v1.Feature(value_type=vt))
        ).result(timeout=600)
        print("  feature", fid)
    except AlreadyExists:
        print("  feature", fid, "exists")

# 4) ingest feature values from the BigQuery table into online + offline storage
ft = Timestamp(); ft.seconds = 1772326500  # fixed feature time for all rows
try:
    fs_client.import_feature_values(
        request=v1.ImportFeatureValuesRequest(
            entity_type=et_path,
            bigquery_source=v1.BigQuerySource(input_uri=BQ_URI),
            entity_id_field="row_id",
            feature_time=ft,
            feature_specs=[v1.ImportFeatureValuesRequest.FeatureSpec(id=fid, source_field=fid) for fid, _ in FEATURES],
            worker_count=1,
        )
    ).result(timeout=2400)
    print("imported feature values")
except Exception as e:
    print("import error:", type(e).__name__, str(e)[:300])

# 5) verify online (low-latency) read for one row
try:
    for rid in ["R00000", "R00001"]:
        resp = online.read_feature_values(request=v1.ReadFeatureValuesRequest(
            entity_type=et_path, entity_id=rid,
            feature_selector=v1.FeatureSelector(id_matcher=v1.IdMatcher(ids=[f for f, _ in FEATURES])),
        ))
        vals = [d for d in resp.entity_view.data]
        print("online read", rid, "->", str(resp.entity_view).replace("\n", " ")[:300])
except Exception as e:
    print("online read error:", type(e).__name__, str(e)[:300])

print("DONE")
