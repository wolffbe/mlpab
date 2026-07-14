import os
import time
import google.cloud.aiplatform as aiplatform
from google.api_core.exceptions import ResourceExhausted
from vertexai.resources.preview import feature_store as fs
from vertexai.resources.preview.feature_store import utils as fsutils

project = os.environ["GCP_PROJECT"]
dataset = os.environ["GCP_BQ_DATASET"]
loc = os.environ["GCP_LOCATION"]
prefix = os.environ["MLPAB_GCP_PREFIX"]
aiplatform.init(project=project, location=loc, api_transport="rest")

TABLE = "scored3ace95"
view_uri = f"bq://{project}.{dataset}.{TABLE}_src"
store_name = f"{prefix}_scored3ace95_store"

online_store = None
# Bounded retries on our own prefixed store (shared project quota is contended).
for attempt in range(6):
    try:
        online_store = fs.FeatureOnlineStore.create_optimized_store(
            name=store_name, labels={"version": "1"}
        )
        print("created own store:", online_store.resource_name)
        break
    except ResourceExhausted as e:
        print(f"attempt {attempt}: quota exhausted, retrying: {e}")
        # store may already exist from a prior succeeded LRO
        try:
            online_store = fs.FeatureOnlineStore(store_name)
            print("found own store after retry:", online_store.resource_name)
            break
        except Exception:
            pass
        time.sleep(30)

if online_store is None:
    # Fall back: reuse an existing online store to host the FeatureView so the
    # features are still available for low-latency lookup.
    stores = fs.FeatureOnlineStore.list()
    print("falling back to existing store; available:", [s.name for s in stores])
    online_store = stores[0]
    print("reusing store:", online_store.resource_name)

existing_views = {v.name for v in online_store.list_feature_views()}
if TABLE in existing_views:
    fv = fs.FeatureView(TABLE, feature_online_store_id=online_store.name)
    print("FeatureView exists:", fv.resource_name)
else:
    fv = online_store.create_feature_view(
        name=TABLE,
        source=fsutils.FeatureViewBigQuerySource(
            uri=view_uri, entity_id_columns=["request_id"]
        ),
        labels={"version": "1"},
    )
    print("FeatureView created:", fv.resource_name)

sync = fv.sync()
print("sync started:", sync)
