import os
from google.cloud import aiplatform_v1 as a

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']; ds = os.environ['GCP_BQ_DATASET']
prefix = os.environ['MLPAB_GCP_PREFIX']
ep = f"{loc}-aiplatform.googleapis.com"
copts = {"api_endpoint": ep}
parent = f"projects/{proj}/locations/{loc}"

admin = a.FeatureOnlineStoreAdminServiceClient(transport="rest", client_options=copts)

store_id = f"{prefix}_airq_fos"
# --- create optimized online store ---
try:
    op = admin.create_feature_online_store(
        parent=parent,
        feature_online_store_id=store_id,
        feature_online_store=a.FeatureOnlineStore(
            optimized=a.FeatureOnlineStore.Optimized()),
    )
    store = op.result(timeout=900)
    print("online store created:", store.name)
except Exception as e:
    print("store create:", type(e).__name__, str(e)[:200])

store_path = f"{parent}/featureOnlineStores/{store_id}"

# --- create feature view over predictions table, entity key = date ---
fv_id = "airqpredf3f1d8"
try:
    op = admin.create_feature_view(
        parent=store_path,
        feature_view_id=fv_id,
        feature_view=a.FeatureView(
            big_query_source=a.FeatureView.BigQuerySource(
                uri=f"bq://{proj}.{ds}.airqpredf3f1d8",
                entity_id_columns=["date"]),
            sync_config=a.FeatureView.SyncConfig(cron="0 * * * *"),
        ),
    )
    fv = op.result(timeout=600)
    print("feature view created:", fv.name)
except Exception as e:
    print("fv create:", type(e).__name__, str(e)[:200])

fv_path = f"{store_path}/featureViews/{fv_id}"

# --- trigger sync so predictions land in online store ---
try:
    resp = admin.sync_feature_view(feature_view=fv_path)
    print("sync triggered:", resp.feature_view_sync)
except Exception as e:
    print("sync:", type(e).__name__, str(e)[:200])
