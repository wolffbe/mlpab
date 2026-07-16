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
MY_STORE = f"{PREFIX}_online_store"
FV = "eventsd3c188"

# 1) Prefer my own store if quota freed up in the meantime.
store = None
try:
    store = fs.FeatureOnlineStore(MY_STORE)
    print("my online store exists:", store.resource_name)
except NotFound:
    try:
        store = fs.FeatureOnlineStore.create_optimized_store(MY_STORE)
        print("created my online store:", store.resource_name)
    except ResourceExhausted:
        print("quota still exhausted for a new store; will host on an existing optimized store")

# 2) Fall back to an existing OPTIMIZED store to provide online serving.
if store is None:
    for s in fs.FeatureOnlineStore.list():
        if s.feature_online_store_type == fs.FeatureOnlineStoreType.OPTIMIZED:
            store = s
            print("using existing optimized store:", store.resource_name)
            break

if store is None:
    raise SystemExit("QUOTA: no optimized online store available")

# Feature view (online low-latency representation, sourced from MY dataset)
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
    print("created feature view:", fv.resource_name)
else:
    print("feature view exists:", fv.resource_name)

sync = fv.sync()
print("sync started:", getattr(sync, "resource_name", sync))
