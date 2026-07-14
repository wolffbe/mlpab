import os
import time
import google.cloud.aiplatform_v1 as v1
from google.api_core.exceptions import AlreadyExists, NotFound, Conflict, ResourceExhausted

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']
parent = "projects/{}/locations/{}".format(proj, loc)
api_endpoint = "{}-aiplatform.googleapis.com".format(loc)
adm = v1.FeatureOnlineStoreAdminServiceClient(client_options={"api_endpoint": api_endpoint}, transport="rest")

FG_ID = "recs75dfad"
OS_ID = "recs75dfad_os"
FV_ID = "recs75dfad"

def make_fv():
    return v1.FeatureView(
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

os_parent = None

# Try to create my own online store, retrying on transient quota churn.
for attempt in range(8):
    try:
        fos = v1.FeatureOnlineStore(
            bigtable=v1.FeatureOnlineStore.Bigtable(
                auto_scaling=v1.FeatureOnlineStore.Bigtable.AutoScaling(
                    min_node_count=1, max_node_count=1, cpu_utilization_target=50
                )
            ),
            labels={"run": os.environ.get('MLPAB_GCP_PREFIX', 'run')},
        )
        op = adm.create_feature_online_store(parent=parent, feature_online_store=fos, feature_online_store_id=OS_ID)
        fos_res = op.result(timeout=1800)
        os_parent = fos_res.name
        print("FeatureOnlineStore created:", os_parent, "state=", fos_res.state)
        break
    except Conflict:
        os_parent = "{}/featureOnlineStores/{}".format(parent, OS_ID)
        print("FeatureOnlineStore already exists (mine):", os_parent)
        break
    except ResourceExhausted:
        print("quota full, retry {}...".format(attempt), flush=True)
        time.sleep(30)

if os_parent is None:
    # Fall back: reuse an existing online store to host the FeatureView.
    stores = list(adm.list_feature_online_stores(parent=parent))
    if not stores:
        raise SystemExit("No online store available and quota exhausted")
    os_parent = stores[0].name
    print("Reusing existing online store:", os_parent)

fv_name = "{}/featureViews/{}".format(os_parent, FV_ID)

# delete stale FV of same id in that store, then create
try:
    adm.get_feature_view(name=fv_name)
    print("Deleting stale FeatureView:", fv_name)
    adm.delete_feature_view(name=fv_name).result(timeout=600)
except NotFound:
    pass

op = adm.create_feature_view(parent=os_parent, feature_view=make_fv(), feature_view_id=FV_ID)
fv_res = op.result(timeout=1800)
print("FeatureView created:", fv_res.name)

# on-demand sync to populate online store
sync_resp = adm.sync_feature_view(feature_view=fv_name)
print("Sync started:", sync_resp.feature_view_sync)
print("OS_PARENT=", os_parent)
