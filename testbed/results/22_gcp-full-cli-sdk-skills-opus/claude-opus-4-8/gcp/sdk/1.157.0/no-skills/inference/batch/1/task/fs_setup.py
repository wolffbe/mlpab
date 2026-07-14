import os
from datetime import datetime, timezone
import google.cloud.aiplatform as aiplatform

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
T = 1773478800000

aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")

FS_ID = f"{PREFIX}_fs"
ENTITY_TYPE_ID = "scores36e30a"   # the feature table name (record key = account_id)
scores_table = f"{PROJECT}.{DATASET}.scores36e30a"

# 1. Featurestore with online serving nodes -> low-latency (online) access enabled.
#    The offline store is populated by ingestion; the online store by the same call.
try:
    fs = aiplatform.Featurestore(FS_ID)
    print("reusing featurestore:", fs.resource_name)
except Exception:
    fs = aiplatform.Featurestore.create(
        featurestore_id=FS_ID,
        online_store_fixed_node_count=1,
        labels={"version": "1"},
        sync=True,
    )
    print("created featurestore:", fs.resource_name)

# 2. Entity type = the feature table; entity id column is the record key account_id.
try:
    et = fs.get_entity_type(ENTITY_TYPE_ID)
    print("reusing entity type:", et.resource_name)
except Exception:
    et = fs.create_entity_type(
        entity_type_id=ENTITY_TYPE_ID,
        description="Batch scores as of T=1773478800000; record key account_id",
        labels={"version": "1"},
        sync=True,
    )
    print("created entity type:", et.resource_name)

# 3. Feature: score (account_id is the entity/record key, not a stored feature).
existing = {f.name for f in et.list_features()}
if "score" not in existing:
    et.create_feature(feature_id="score", value_type="DOUBLE", description="logistic score", sync=True)
    print("created feature: score")
else:
    print("feature score already exists")

# 4. Ingest from BigQuery -> populates offline AND online stores.
feature_time = datetime.fromtimestamp(T / 1000.0, tz=timezone.utc).replace(tzinfo=None)
et.ingest_from_bq(
    feature_ids=["score"],
    feature_time=feature_time,
    bq_source_uri=f"bq://{scores_table}",
    entity_id_field="account_id",
    disable_online_serving=False,
    sync=True,
)
print("ingestion complete")

print("FEATURESTORE:", fs.resource_name)
print("ENTITY_TYPE:", et.resource_name)
print("FEATURES:", [f.name for f in et.list_features()])
