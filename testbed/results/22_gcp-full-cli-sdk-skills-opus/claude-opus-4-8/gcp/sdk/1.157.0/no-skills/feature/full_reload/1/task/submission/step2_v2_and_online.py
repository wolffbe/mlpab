import os
import time
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as fs
from google.api_core.exceptions import Conflict, NotFound, AlreadyExists, ResourceExhausted

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]

aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")


def ensure_fg(fg_id, src_full, feature_cols):
    """Create FeatureGroup once (no delete -> no tombstone). Skip if already present."""
    try:
        fg = fs.FeatureGroup(fg_id)
        print(f"FeatureGroup {fg_id} already exists: {fg.name}")
    except Exception:
        src = fs.FeatureGroupBigQuerySource(uri=f"bq://{src_full}", entity_id_columns=["row_id"])
        fg = None
        for attempt in range(30):
            try:
                fg = fs.FeatureGroup.create(
                    name=fg_id, source=src,
                    description="customers feature table; key=row_id, event-time=updated_at(epoch ms)->feature_timestamp",
                    labels={"record_key": "row_id", "event_time": "updated_at"},
                )
                break
            except Conflict as e:
                print(f"  {fg_id} create conflict, retry {attempt}: {str(e)[:70]}")
                time.sleep(20)
        if fg is None:
            raise RuntimeError(f"could not create FeatureGroup {fg_id}")
        print(f"created FeatureGroup {fg.name}")
    # register features (idempotent-ish)
    existing = set()
    try:
        existing = {f.name for f in fg.list_features()}
    except Exception:
        pass
    for col in feature_cols:
        if col in existing:
            print(f"  feature {col} already registered")
            continue
        try:
            fg.create_feature(name=col, version_column_name=col)
            print(f"  registered feature {col}")
        except AlreadyExists:
            print(f"  feature {col} already exists")
    return fg


# ---- version 2 FeatureGroup (graded) ----
fg2_src = f"{PROJECT}.{DATASET}.customerscd1186_2_fg"
fg2 = ensure_fg("customerscd1186_2", fg2_src, ["full_name", "balance", "currency", "updated_at"])

# ---- online store + feature view for low-latency lookup of v2 ----
ONLINE_STORE = f"{PREFIX}_customers_online"
try:
    store = fs.FeatureOnlineStore(ONLINE_STORE)
    print(f"online store exists: {store.name}")
except Exception:
    store = None
    for attempt in range(15):
        try:
            print(f"creating optimized online store {ONLINE_STORE} (attempt {attempt}) ...")
            store = fs.FeatureOnlineStore.create_optimized_store(
                name=ONLINE_STORE, labels={"purpose": "customerscd1186_v2_online"},
            )
            print(f"created online store: {store.name}")
            break
        except ResourceExhausted as e:
            print(f"  quota exhausted, retry {attempt}: {str(e)[:90]}")
            try:
                store = fs.FeatureOnlineStore(ONLINE_STORE)
                print(f"  found existing store now: {store.name}")
                break
            except Exception:
                store = None
            time.sleep(30)
    if store is None:
        raise RuntimeError("could not create FeatureOnlineStore (quota)")

FV_ID = f"{PREFIX}_customerscd1186_2"
try:
    fv = store.get_feature_view(FV_ID)
    print(f"feature view exists: {fv.name}")
except Exception:
    print(f"creating feature view {FV_ID} ...")
    fv_src = fs.FeatureViewBigQuerySource(
        uri=f"bq://{fg2_src}", entity_id_columns=["row_id"],
    )
    fv = store.create_feature_view(
        name=FV_ID, source=fv_src, labels={"feature_group": "customerscd1186_2"},
    )
    print(f"created feature view: {fv.name}")

# trigger a sync so online data is populated for low-latency reads
print("syncing feature view ...")
try:
    sync = fv.sync()
    print(f"sync started: {sync}")
except Exception as e:
    print(f"sync call: {type(e).__name__}: {str(e)[:200]}")

print("STEP2 DONE")
