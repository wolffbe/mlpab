import os, time
from google.cloud import aiplatform_v1 as v1

PROJECT = os.environ["GCP_PROJECT"]
DATASET = os.environ["GCP_BQ_DATASET"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]

ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"
PARENT = f"projects/{PROJECT}/locations/{LOCATION}"
STORE_ID = f"{PREFIX}_features347afc_store"
FV_ID = "features347afc"
BQ_URI = f"bq://{PROJECT}.{DATASET}.features347afc"

admin = v1.FeatureOnlineStoreAdminServiceClient(
    transport="rest", client_options={"api_endpoint": ENDPOINT})

# --- 1. Bigtable online store (low-latency serving) ---
store_res = f"{PARENT}/featureOnlineStores/{STORE_ID}"
existing = {s.name.split("/")[-1] for s in admin.list_feature_online_stores(parent=PARENT)}
if STORE_ID in existing:
    print("reusing store:", store_res)
else:
    fos = v1.FeatureOnlineStore(
        bigtable=v1.FeatureOnlineStore.Bigtable(
            auto_scaling=v1.FeatureOnlineStore.Bigtable.AutoScaling(
                min_node_count=1, max_node_count=1, cpu_utilization_target=50)))
    op = admin.create_feature_online_store(
        parent=PARENT, feature_online_store_id=STORE_ID, feature_online_store=fos)
    print("creating store ...")
    op.result(timeout=1200)
    print("created store:", store_res)

# --- 2. Feature view over BigQuery offline table, keyed by row_id ---
fv_res = f"{store_res}/featureViews/{FV_ID}"
existing_fv = {v.name.split("/")[-1] for v in admin.list_feature_views(parent=store_res)}
if FV_ID in existing_fv:
    print("reusing feature view:", fv_res)
else:
    fv = v1.FeatureView(
        big_query_source=v1.FeatureView.BigQuerySource(
            uri=BQ_URI, entity_id_columns=["row_id"]))
    op = admin.create_feature_view(
        parent=store_res, feature_view_id=FV_ID, feature_view=fv)
    op.result(timeout=600)
    print("created feature view:", fv_res)

# --- 3. Sync BigQuery data into the online store ---
resp = admin.sync_feature_view(feature_view=fv_res)
sync_name = resp.feature_view_sync
print("started sync:", sync_name)
deadline = time.time() + 1500
done = False
while time.time() < deadline:
    s = admin.get_feature_view_sync(name=sync_name)
    end = s.run_time.end_time
    if end and end.seconds:
        done = True
        print("sync end_time:", end)
        break
    time.sleep(20)
print("sync complete:", done)
