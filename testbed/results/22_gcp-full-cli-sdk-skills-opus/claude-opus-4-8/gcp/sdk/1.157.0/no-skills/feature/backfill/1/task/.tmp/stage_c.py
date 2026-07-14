import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as fs
from google.api_core.exceptions import ResourceExhausted

PROJECT = os.environ["GCP_PROJECT"]; LOCATION = os.environ["GCP_LOCATION"]; DATASET = os.environ["GCP_BQ_DATASET"]
aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")

BQ_URI = f"bq://{PROJECT}.{DATASET}.accountsed4daa"
STORE_NAME = "accountsed4daa_store"
FV_NAME = "accountsed4daa"

# --- get or create online store ---
store = None
try:
    store = fs.FeatureOnlineStore(STORE_NAME)
    print("Online store already exists:", store.resource_name)
except Exception:
    try:
        store = fs.FeatureOnlineStore.create_optimized_store(STORE_NAME, labels={"version": "1"})
        print("Created online store:", store.resource_name)
    except ResourceExhausted as e:
        print("QUOTA_EXHAUSTED:", e)
        raise SystemExit(3)

# --- get or create feature view from the BigQuery source ---
try:
    fv = fs.FeatureView(FV_NAME, feature_online_store_id=STORE_NAME)
    print("FeatureView already exists:", fv.resource_name)
except Exception:
    fv = store.create_feature_view(
        FV_NAME,
        source=fs.FeatureViewBigQuerySource(uri=BQ_URI, entity_id_columns=["row_id"]),
        labels={"version": "1"},
    )
    print("Created FeatureView:", fv.resource_name)

# --- trigger a one-time sync so latest values are loaded into the online store ---
sync_obj = fv.sync()
print("Started sync:", sync_obj.resource_name if hasattr(sync_obj, "resource_name") else sync_obj)
print("STAGE_C_DONE")
