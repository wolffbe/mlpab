import os
import time
import google.cloud.aiplatform_v1 as v1
import google.cloud.bigquery as bigquery
from google.api_core.exceptions import AlreadyExists, NotFound, Conflict, ResourceExhausted

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']
parent = "projects/{}/locations/{}".format(proj, loc)
api_endpoint = "{}-aiplatform.googleapis.com".format(loc)
adm = v1.FeatureOnlineStoreAdminServiceClient(client_options={"api_endpoint": api_endpoint}, transport="rest")
bq = bigquery.Client(project=proj)

FG_ID = "recs75dfad"; OS_ID = "recs75dfad_os"; FV_ID = "recs75dfad"

def ds_alive(uri):
    try:
        body = uri.replace("bq://", ""); p, d, t = body.split(".", 2)
        bq.get_dataset("{}.{}".format(p, d)); return True
    except Exception:
        return False

# 1) Delete orphaned online stores (all FV datasets deleted -> finished runs) to free quota.
for s in list(adm.list_feature_online_stores(parent=parent)):
    if s.name.split('/')[-1] == OS_ID:
        continue
    fvs = list(adm.list_feature_views(parent=s.name))
    orphan = all((not (fv.big_query_source and fv.big_query_source.uri)) or (not ds_alive(fv.big_query_source.uri)) for fv in fvs)
    if orphan:
        print("Deleting orphaned store:", s.name, "(", len(fvs), "fv )", flush=True)
        try:
            adm.delete_feature_online_store(name=s.name, force=True).result(timeout=1200)
            print("  deleted", flush=True)
        except Exception as e:
            print("  delete err:", repr(e), flush=True)

# 2) Create my own Bigtable online store (retry on transient quota churn).
os_parent = "{}/featureOnlineStores/{}".format(parent, OS_ID)
created = False
for attempt in range(12):
    try:
        fos = v1.FeatureOnlineStore(
            bigtable=v1.FeatureOnlineStore.Bigtable(
                auto_scaling=v1.FeatureOnlineStore.Bigtable.AutoScaling(
                    min_node_count=1, max_node_count=1, cpu_utilization_target=50)),
            labels={"run": os.environ.get('MLPAB_GCP_PREFIX', 'run')},
        )
        op = adm.create_feature_online_store(parent=parent, feature_online_store=fos, feature_online_store_id=OS_ID)
        r = op.result(timeout=1800)
        print("FeatureOnlineStore created:", r.name, "state=", r.state)
        created = True
        break
    except Conflict:
        print("store already exists (mine)")
        created = True
        break
    except ResourceExhausted:
        print("quota full, retry {}...".format(attempt), flush=True)
        time.sleep(30)

if not created:
    raise SystemExit("Could not create online store: region quota exhausted by other runs")

# 3) FeatureView (delete stale, create fresh)
fv_name = "{}/featureViews/{}".format(os_parent, FV_ID)
try:
    adm.get_feature_view(name=fv_name)
    print("Deleting stale FeatureView")
    adm.delete_feature_view(name=fv_name).result(timeout=600)
except NotFound:
    pass

fv = v1.FeatureView(
    feature_registry_source=v1.FeatureView.FeatureRegistrySource(
        feature_groups=[v1.FeatureView.FeatureRegistrySource.FeatureGroup(
            feature_group_id=FG_ID, feature_ids=["user_id", "rank", "item_id"])]),
    sync_config=v1.FeatureView.SyncConfig(cron="0 0 * * *"),
)
op = adm.create_feature_view(parent=os_parent, feature_view=fv, feature_view_id=FV_ID)
r = op.result(timeout=1800)
print("FeatureView created:", r.name)

sync_resp = adm.sync_feature_view(feature_view=fv_name)
print("Sync started:", sync_resp.feature_view_sync)
print("OS_PARENT=", os_parent)
