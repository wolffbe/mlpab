import os, time
import google.cloud.aiplatform_v1 as v1

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']
parent = "projects/{}/locations/{}".format(proj, loc)
api_endpoint = "{}-aiplatform.googleapis.com".format(loc)
adm = v1.FeatureOnlineStoreAdminServiceClient(client_options={"api_endpoint": api_endpoint}, transport="rest")

OS_ID = "recs75dfad_os"; FV_ID = "recs75dfad"
os_name = "{}/featureOnlineStores/{}".format(parent, OS_ID)
fv_name = "{}/featureViews/{}".format(os_name, FV_ID)

store = adm.get_feature_online_store(name=os_name)
print("store state:", store.state)
fv = adm.get_feature_view(name=fv_name)
print("fv name:", fv.name)

resp = adm.sync_feature_view(feature_view=fv_name)
print("triggered sync:", resp.feature_view_sync, flush=True)

for i in range(40):
    syncs = list(adm.list_feature_view_syncs(parent=fv_name))
    if syncs:
        s = sorted(syncs, key=lambda x: x.create_time.timestamp() if x.create_time else 0)[-1]
        end = s.run_time.end_time if s.run_time else None
        code = s.final_status.code if s.final_status else None
        rows = getattr(s.sync_summary, 'row_synced', None)
        print("sync", s.name.split('/')[-1], "end_set=", bool(end and end.seconds), "code=", code, "rows=", rows, flush=True)
        if end and end.seconds:
            print("SYNC COMPLETE rows=", rows)
            break
    else:
        print("no syncs listed yet", flush=True)
    time.sleep(20)
