import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as fs

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]

aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")

BQ_URI = f"bq://{PROJECT}.{DATASET}.accountsed4daa"
FG_NAME = "accountsed4daa"
STORE_NAME = "accountsed4daa_store"

# ---- FeatureGroup (the registered feature table on top of BigQuery) ----
try:
    fg = fs.FeatureGroup(FG_NAME)
    print("FeatureGroup exists:", fg.resource_name)
except Exception:
    fg = fs.FeatureGroup.create(
        FG_NAME,
        source=fs.FeatureGroupBigQuerySource(uri=BQ_URI, entity_id_columns=["row_id"]),
        description="accounts feature table v1; record key row_id, event-time updated_at (epoch ms)",
        labels={"version": "1"},
    )
    print("Created FeatureGroup:", fg.resource_name)

# ---- Register features (event-time column updated_at + the two features) ----
existing = {f.name for f in fg.list_features()}
print("existing features:", existing)
for col in ["status", "balance", "updated_at"]:
    if col not in existing:
        fg.create_feature(col, version_column_name=col)
        print("registered feature:", col)
print("features now:", [f.name for f in fg.list_features()])

# ---- Online store (low-latency serving layer) ----
try:
    store = fs.FeatureOnlineStore(STORE_NAME)
    print("Online store exists:", store.resource_name)
except Exception:
    store = fs.FeatureOnlineStore.create_optimized_store(
        STORE_NAME, labels={"version": "1"}
    )
    print("Created online store:", store.resource_name)
print("store type:", store.feature_online_store_type)
print("STAGE_B_DONE")
