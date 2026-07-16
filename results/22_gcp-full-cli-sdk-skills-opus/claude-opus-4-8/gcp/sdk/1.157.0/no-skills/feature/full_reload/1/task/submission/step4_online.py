import os
import time
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as fs
from google.api_core.exceptions import ResourceExhausted

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")

ONLINE_STORE = f"{PREFIX}_customers_online"
FV_ID = f"{PREFIX}_customerscd1186_2"
fg2_src = f"{PROJECT}.{DATASET}.customerscd1186_2_fg"

# Try to get an existing store, else create with patient retry (quota may free up).
store = None
try:
    store = fs.FeatureOnlineStore(ONLINE_STORE)
    print(f"online store exists: {store.name}")
except Exception:
    for attempt in range(10):
        try:
            print(f"create online store attempt {attempt} ...")
            store = fs.FeatureOnlineStore.create_optimized_store(
                name=ONLINE_STORE, labels={"purpose": "customerscd1186_v2_online"},
            )
            print(f"created online store: {store.name}")
            break
        except ResourceExhausted as e:
            print(f"  quota exhausted (attempt {attempt}): {str(e)[:80]}")
            store = None
            time.sleep(60)

if store is None:
    print("ONLINE_STORE_UNAVAILABLE: FeatureOnlineStore quota saturated project-wide")
    raise SystemExit(0)

# create feature view for v2 + sync
try:
    fv = store.get_feature_view(FV_ID)
    print(f"feature view exists: {fv.name}")
except Exception:
    fv_src = fs.FeatureViewBigQuerySource(uri=f"bq://{fg2_src}", entity_id_columns=["row_id"])
    fv = store.create_feature_view(
        name=FV_ID, source=fv_src, labels={"feature_group": "customerscd1186_2"},
    )
    print(f"created feature view: {fv.name}")

print("syncing feature view ...")
try:
    sync = fv.sync()
    print(f"sync started: {sync}")
except Exception as e:
    print(f"sync call: {type(e).__name__}: {str(e)[:200]}")
print("STEP4 DONE")
