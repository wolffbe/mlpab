import os, time
import google.cloud.aiplatform_v1 as v1

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]

API = f"{LOCATION}-aiplatform.googleapis.com"
parent = f"projects/{PROJECT}/locations/{LOCATION}"
BQ_URI = f"bq://{PROJECT}.{DATASET}.features347afc"

FG_ID = "features347afc"
FV_ID = "features347afc"
FOS_ID = f"{PREFIX}_fos"
FEATURE_COLS = ["account_id", "event_time", "amount_usd", "is_weekend", "amount_7d"]

reg = v1.FeatureRegistryServiceClient(client_options={"api_endpoint": API}, transport="rest")
admin = v1.FeatureOnlineStoreAdminServiceClient(client_options={"api_endpoint": API}, transport="rest")

# ---------- 1. FeatureGroup (offline registry, record key = row_id) ----------
fg = v1.FeatureGroup(
    big_query=v1.FeatureGroup.BigQuery(
        big_query_source=v1.BigQuerySource(input_uri=BQ_URI),
        entity_id_columns=["row_id"],
        static_data_source=True,
    ),
    description="Derived transaction features (v1); record key row_id, event_time epoch ms",
)
try:
    reg.create_feature_group(
        request=v1.CreateFeatureGroupRequest(parent=parent, feature_group=fg, feature_group_id=FG_ID)
    ).result(timeout=600)
    print("created feature group", FG_ID)
except Exception as e:
    print("feature group:", type(e).__name__, str(e)[:200])

fg_path = reg.feature_group_path(PROJECT, LOCATION, FG_ID)

# register each feature column
for col in FEATURE_COLS:
    try:
        reg.create_feature(
            request=v1.CreateFeatureRequest(
                parent=fg_path,
                feature=v1.Feature(version_column_name=col),
                feature_id=col,
            )
        ).result(timeout=600)
        print("  feature", col)
    except Exception as e:
        print("  feature", col, ":", type(e).__name__, str(e)[:120])

# ---------- 2. FeatureOnlineStore (low-latency serving container) ----------
fos = v1.FeatureOnlineStore(optimized=v1.FeatureOnlineStore.Optimized())
try:
    admin.create_feature_online_store(
        request=v1.CreateFeatureOnlineStoreRequest(
            parent=parent, feature_online_store=fos, feature_online_store_id=FOS_ID
        )
    ).result(timeout=1800)
    print("created online store", FOS_ID)
except Exception as e:
    print("online store:", type(e).__name__, str(e)[:200])

fos_path = admin.feature_online_store_path(PROJECT, LOCATION, FOS_ID)

# ---------- 3. FeatureView (online view over the registered feature group) ----------
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
try:
    admin.create_feature_view(
        request=v1.CreateFeatureViewRequest(
            parent=fos_path, feature_view=fv, feature_view_id=FV_ID
        )
    ).result(timeout=1800)
    print("created feature view", FV_ID)
except Exception as e:
    print("feature view:", type(e).__name__, str(e)[:200])

fv_path = admin.feature_view_path(PROJECT, LOCATION, FOS_ID, FV_ID)

# ---------- 4. Trigger an immediate sync into the online store ----------
try:
    resp = admin.sync_feature_view(feature_view=fv_path)
    sync_name = resp.feature_view_sync
    print("triggered sync:", sync_name)
    for _ in range(60):
        s = admin.get_feature_view_sync(name=sync_name)
        end = s.run_time.end_time.seconds if s.run_time else 0
        if end:
            print("sync finished; rows synced:",
                  getattr(s.sync_summary, "row_synced", "n/a"))
            break
        time.sleep(15)
    else:
        print("sync still running after wait window")
except Exception as e:
    print("sync:", type(e).__name__, str(e)[:200])

print("DONE")
