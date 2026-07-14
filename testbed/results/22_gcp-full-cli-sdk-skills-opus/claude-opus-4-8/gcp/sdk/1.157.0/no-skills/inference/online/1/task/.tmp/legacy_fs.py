import os
import google.cloud.aiplatform as aiplatform
from google.cloud.aiplatform import Featurestore, EntityType

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")

FS_ID = f"{PREFIX}_profilesaf22bf_fs"
ET_ID = "profilesaf22bf"          # the feature table (record key: account_id)
FEATURES = ["f1", "f2", "f3", "f4"]
BQ_URI = f"bq://{PROJECT}.{DATASET}.profilesaf22bf"

# 1. Featurestore with online serving enabled (low-latency access)
try:
    fs = Featurestore(FS_ID)
    print("Featurestore exists:", fs.resource_name)
except Exception:
    fs = Featurestore.create(FS_ID, online_store_fixed_node_count=1)
    print("Created Featurestore:", fs.resource_name)

# 2. EntityType = feature table, record key account_id
try:
    et = fs.get_entity_type(ET_ID)
    print("EntityType exists:", et.resource_name)
except Exception:
    et = fs.create_entity_type(ET_ID, description="account feature profiles v1")
    print("Created EntityType:", et.resource_name)

# 3. Features f1..f4 (DOUBLE)
existing = {f.name for f in et.list_features()}
for feat in FEATURES:
    if feat in existing:
        print("feature exists:", feat); continue
    et.create_feature(feat, value_type="DOUBLE")
    print("created feature:", feat)

# 4. Ingest from BigQuery into online + offline store
et.ingest_from_bq(
    feature_ids=FEATURES,
    feature_time="feature_timestamp",
    bq_source_uri=BQ_URI,
    entity_id_field="account_id",
)
print("INGEST_DONE")
