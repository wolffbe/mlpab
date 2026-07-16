import os, time
import vertexai
from vertexai.resources import preview as fs
from vertexai.resources.preview.feature_store import utils

PROJECT = os.environ["GCP_PROJECT"]
DATASET = os.environ["GCP_BQ_DATASET"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]

vertexai.init(project=PROJECT, location=LOCATION, api_transport="rest")
BQ_URI = f"bq://{PROJECT}.{DATASET}.eventsd3c188"
STORE = f"{PREFIX}_online_store"
FV = "eventsd3c188"

# ---- Online store (optimized = low-latency real-time serving) ----
store = None
try:
    store = fs.FeatureOnlineStore(STORE)
    print("Online store exists:", store.resource_name)
except Exception:
    store = fs.FeatureOnlineStore.create_optimized_store(STORE)
    print("Created online store:", store.resource_name)

# ---- Feature view (online representation of the feature table) ----
fv = None
for existing_fv in store.list_feature_views():
    if existing_fv.name == FV:
        fv = existing_fv
        break
if fv is None:
    fv = store.create_feature_view(
        FV,
        source=utils.FeatureViewBigQuerySource(uri=BQ_URI, entity_id_columns=["row_id"]),
        sync_config="0 0 * * *",  # daily; we also trigger an immediate sync below
    )
    print("Created feature view:", fv.resource_name)
else:
    print("Feature view exists:", fv.resource_name)

# ---- Trigger sync so online data is populated for low-latency lookup ----
sync = fv.sync()
print("sync started:", sync.resource_name if hasattr(sync, "resource_name") else sync)
