import os, time
import vertexai
from vertexai.resources import preview as fs
from vertexai.resources.preview.feature_store import utils
from google.api_core.exceptions import ResourceExhausted, NotFound

PROJECT = os.environ["GCP_PROJECT"]
DATASET = os.environ["GCP_BQ_DATASET"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]

vertexai.init(project=PROJECT, location=LOCATION, api_transport="rest")
BQ_URI = f"bq://{PROJECT}.{DATASET}.eventsd3c188"
STORE = f"{PREFIX}_online_store"
FV = "eventsd3c188"

store = None
try:
    store = fs.FeatureOnlineStore(STORE)
    print("Online store exists:", store.resource_name)
except NotFound:
    pass

if store is None:
    for attempt in range(20):
        try:
            store = fs.FeatureOnlineStore.create_optimized_store(STORE)
            print("Created online store:", store.resource_name)
            break
        except ResourceExhausted:
            print(f"attempt {attempt}: FeatureOnlineStores quota exhausted, retrying...")
            time.sleep(30)

if store is None:
    raise SystemExit("QUOTA: could not create optimized online store after retries")

# Feature view
fv = None
for e in store.list_feature_views():
    if e.name == FV:
        fv = e
        break
if fv is None:
    fv = store.create_feature_view(
        FV,
        source=utils.FeatureViewBigQuerySource(uri=BQ_URI, entity_id_columns=["row_id"]),
    )
    print("Created feature view:", fv.resource_name)
else:
    print("Feature view exists:", fv.resource_name)

sync = fv.sync()
print("sync started:", getattr(sync, "resource_name", sync))
