import os, time
import google.cloud.aiplatform_v1 as v1
from google.api_core.exceptions import AlreadyExists, ResourceExhausted

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
API = f"{LOCATION}-aiplatform.googleapis.com"
parent = f"projects/{PROJECT}/locations/{LOCATION}"

FG_ID = "features347afc"
FV_ID = "features347afc"
FOS_ID = f"{PREFIX}_fos"
FEATURE_COLS = ["account_id", "event_time", "amount_usd", "is_weekend", "amount_7d"]

admin = v1.FeatureOnlineStoreAdminServiceClient(client_options={"api_endpoint": API}, transport="rest")

fos_path = admin.feature_online_store_path(PROJECT, LOCATION, FOS_ID)
fv_path = admin.feature_view_path(PROJECT, LOCATION, FOS_ID, FV_ID)

# ---- create online store, retrying on quota ----
def store_exists():
    try:
        admin.get_feature_online_store(name=fos_path)
        return True
    except Exception:
        return False

made = store_exists()
attempts = 0
while not made and attempts < 24:
    attempts += 1
    try:
        fos = v1.FeatureOnlineStore(optimized=v1.FeatureOnlineStore.Optimized())
        admin.create_feature_online_store(
            request=v1.CreateFeatureOnlineStoreRequest(
                parent=parent, feature_online_store=fos, feature_online_store_id=FOS_ID
            )
        ).result(timeout=1800)
        print("created online store", FOS_ID)
        made = True
    except AlreadyExists:
        made = True
        print("online store already exists")
    except ResourceExhausted as e:
        print(f"attempt {attempts}: quota exhausted, waiting 60s...", flush=True)
        time.sleep(60)
    except Exception as e:
        print("online store error:", type(e).__name__, str(e)[:200])
        break

if not made:
    print("ONLINE_STORE_UNAVAILABLE: FeatureOnlineStores quota exhausted after retries")
    raise SystemExit(0)

# ---- feature view ----
try:
    fv = v1.FeatureView(
        feature_registry_source=v1.FeatureView.FeatureRegistrySource(
            feature_groups=[
                v1.FeatureView.FeatureRegistrySource.FeatureGroup(
                    feature_group_id=FG_ID, feature_ids=FEATURE_COLS
                )
            ]
        ),
        sync_config=v1.FeatureView.SyncConfig(cron="0 * * * *"),
    )
    admin.create_feature_view(
        request=v1.CreateFeatureViewRequest(parent=fos_path, feature_view=fv, feature_view_id=FV_ID)
    ).result(timeout=1800)
    print("created feature view", FV_ID)
except AlreadyExists:
    print("feature view already exists")
except Exception as e:
    print("feature view error:", type(e).__name__, str(e)[:300])

# ---- trigger sync ----
try:
    resp = admin.sync_feature_view(feature_view=fv_path)
    sync_name = resp.feature_view_sync
    print("triggered sync:", sync_name)
    for _ in range(80):
        s = admin.get_feature_view_sync(name=sync_name)
        end = s.run_time.end_time.seconds if s.run_time else 0
        if end:
            print("sync finished; summary:", str(s.sync_summary).replace("\n", " "))
            break
        time.sleep(15)
    else:
        print("sync still running after wait window")
except Exception as e:
    print("sync error:", type(e).__name__, str(e)[:300])

print("DONE")
