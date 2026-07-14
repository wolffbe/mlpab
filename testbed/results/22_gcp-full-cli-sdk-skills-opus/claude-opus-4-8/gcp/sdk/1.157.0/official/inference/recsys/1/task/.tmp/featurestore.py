import os
import time
import google.cloud.aiplatform_v1 as v1
from google.api_core.exceptions import AlreadyExists, NotFound, Conflict

proj = os.environ['GCP_PROJECT']
loc = os.environ['GCP_LOCATION']
ds = os.environ['GCP_BQ_DATASET']
parent = "projects/{}/locations/{}".format(proj, loc)
api_endpoint = "{}-aiplatform.googleapis.com".format(loc)

FG_ID = "recs75dfad"
OS_ID = "recs75dfad_os"
FV_ID = "recs75dfad"
bq_uri = "bq://{}.{}.recs75dfad".format(proj, ds)

reg = v1.FeatureRegistryServiceClient(client_options={"api_endpoint": api_endpoint}, transport="rest")
adm = v1.FeatureOnlineStoreAdminServiceClient(client_options={"api_endpoint": api_endpoint}, transport="rest")

fg_parent = "{}/featureGroups/{}".format(parent, FG_ID)
os_parent = "{}/featureOnlineStores/{}".format(parent, OS_ID)
fv_name = "{}/featureViews/{}".format(os_parent, FV_ID)

# 1) FeatureGroup (offline, BigQuery-backed) -- delete any stale one first
try:
    existing = reg.get_feature_group(name=fg_parent)
    print("Deleting stale FeatureGroup:", existing.name)
    reg.delete_feature_group(name=fg_parent, force=True).result(timeout=600)
except NotFound:
    pass

fg = v1.FeatureGroup(
    big_query=v1.FeatureGroup.BigQuery(
        big_query_source=v1.BigQuerySource(input_uri=bq_uri),
        entity_id_columns=["rec_id"],
        static_data_source=True,
    ),
    description="Top-5 recommendations per user (two-tower dot product).",
)
fg_res = None
for attempt in range(40):
    try:
        op = reg.create_feature_group(parent=parent, feature_group=fg, feature_group_id=FG_ID)
        fg_res = op.result(timeout=600)
        break
    except Conflict as e:
        print("waiting for stale delete to settle (attempt {})...".format(attempt), flush=True)
        time.sleep(30)
if fg_res is None:
    raise SystemExit("FeatureGroup create kept conflicting")
print("FeatureGroup created:", fg_res.name)

# 2) Features (non-entity columns)
for feat_id in ["user_id", "rank", "item_id"]:
    try:
        fop = reg.create_feature(parent=fg_parent, feature=v1.Feature(description="col " + feat_id), feature_id=feat_id)
        fr = fop.result(timeout=600)
        print("Feature created:", fr.name)
    except (AlreadyExists, Conflict):
        print("Feature exists:", feat_id)

# 3) FeatureOnlineStore (online, Bigtable-backed for low-latency lookup)
try:
    fos = v1.FeatureOnlineStore(
        bigtable=v1.FeatureOnlineStore.Bigtable(
            auto_scaling=v1.FeatureOnlineStore.Bigtable.AutoScaling(
                min_node_count=1, max_node_count=1, cpu_utilization_target=50
            )
        ),
        labels={"run": os.environ.get('MLPAB_GCP_PREFIX', 'run')},
    )
    op = adm.create_feature_online_store(
        parent=parent, feature_online_store=fos, feature_online_store_id=OS_ID
    )
    fos_res = op.result(timeout=1800)
    print("FeatureOnlineStore created:", fos_res.name, "state=", fos_res.state)
except (AlreadyExists, Conflict):
    print("FeatureOnlineStore exists:", adm.get_feature_online_store(name=os_parent).name)

# 4) FeatureView referencing the FeatureGroup + features (online serving)
try:
    adm.get_feature_view(name=fv_name)
    print("Deleting stale FeatureView:", fv_name)
    adm.delete_feature_view(name=fv_name).result(timeout=600)
except NotFound:
    pass

fv = v1.FeatureView(
    feature_registry_source=v1.FeatureView.FeatureRegistrySource(
        feature_groups=[
            v1.FeatureView.FeatureRegistrySource.FeatureGroup(
                feature_group_id=FG_ID,
                feature_ids=["user_id", "rank", "item_id"],
            )
        ]
    ),
    sync_config=v1.FeatureView.SyncConfig(cron="0 0 * * *"),
)
op = adm.create_feature_view(parent=os_parent, feature_view=fv, feature_view_id=FV_ID)
fv_res = op.result(timeout=1800)
print("FeatureView created:", fv_res.name)

# 5) Trigger an on-demand sync to populate the online store
sync_resp = adm.sync_feature_view(feature_view=fv_name)
print("Sync started:", sync_resp.feature_view_sync)
