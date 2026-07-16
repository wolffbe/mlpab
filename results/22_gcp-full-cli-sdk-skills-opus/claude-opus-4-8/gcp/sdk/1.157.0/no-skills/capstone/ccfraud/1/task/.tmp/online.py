from google.cloud import aiplatform_v1 as v1
import os, time
proj=os.environ['GCP_PROJECT']; loc=os.environ['GCP_LOCATION']; ds=os.environ['GCP_BQ_DATASET']; pref=os.environ['MLPAB_GCP_PREFIX']
ep=f"{loc}-aiplatform.googleapis.com"
adm=v1.FeatureOnlineStoreAdminServiceClient(client_options={"api_endpoint":ep}, transport="rest")
parent=f"projects/{proj}/locations/{loc}"
FOS_ID=f"{pref}_ccfos"
FV_ID=f"{pref}_ccpred76ccb2"

# 1. Create Bigtable-backed online store
fos=v1.FeatureOnlineStore(
    bigtable=v1.FeatureOnlineStore.Bigtable(
        auto_scaling=v1.FeatureOnlineStore.Bigtable.AutoScaling(min_node_count=1, max_node_count=1)),
    labels={"mlpab_prefix":pref, "task":"fraud"})
try:
    op=adm.create_feature_online_store(parent=parent, feature_online_store=fos, feature_online_store_id=FOS_ID)
    print("creating online store...")
    r=op.result(timeout=1200)
    print("online store:", r.name)
except Exception as e:
    print("FOS create note:", type(e).__name__, str(e)[:200])
    print("existing:", adm.get_feature_online_store(name=adm.feature_online_store_path(proj,loc,FOS_ID)).name)

fos_path=adm.feature_online_store_path(proj,loc,FOS_ID)

# 2. Create FeatureView from BQ predictions table, keyed by transaction_id
fv=v1.FeatureView(
    big_query_source=v1.FeatureView.BigQuerySource(
        uri=f"bq://{proj}.{ds}.ccpred76ccb2",
        entity_id_columns=["transaction_id"]),
    sync_config=v1.FeatureView.SyncConfig(cron="0 0 * * *"),
    labels={"mlpab_prefix":pref, "task":"fraud"})
try:
    op2=adm.create_feature_view(parent=fos_path, feature_view=fv, feature_view_id=FV_ID)
    r2=op2.result(timeout=600)
    print("feature view:", r2.name)
except Exception as e:
    print("FV create note:", type(e).__name__, str(e)[:200])

fv_path=adm.feature_view_path(proj,loc,FOS_ID,FV_ID)
# 3. Trigger immediate sync to load online data
try:
    s=adm.sync_feature_view(feature_view=fv_path)
    print("sync started:", s.feature_view_sync)
except Exception as e:
    print("sync note:", type(e).__name__, str(e)[:200])
print("done")
