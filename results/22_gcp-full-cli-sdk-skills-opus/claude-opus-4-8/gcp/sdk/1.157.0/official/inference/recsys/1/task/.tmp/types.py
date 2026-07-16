import os
import google.cloud.aiplatform_v1 as v1
import google.cloud.bigquery as bigquery

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']
parent = "projects/{}/locations/{}".format(proj, loc)
api_endpoint = "{}-aiplatform.googleapis.com".format(loc)
adm = v1.FeatureOnlineStoreAdminServiceClient(client_options={"api_endpoint": api_endpoint}, transport="rest")
bq = bigquery.Client(project=proj)

def ds_alive(uri):
    try:
        body = uri.replace("bq://", ""); p, d, t = body.split(".", 2)
        bq.get_dataset("{}.{}".format(p, d)); return True
    except Exception:
        return False

for s in adm.list_feature_online_stores(parent=parent):
    which = s._pb.WhichOneof("storage_type")
    fvs = list(adm.list_feature_views(parent=s.name))
    alive = []
    for fv in fvs:
        uri = fv.big_query_source.uri if fv.big_query_source else ""
        alive.append(ds_alive(uri) if uri else True)
    print(s.name.split('/')[-1], "storage=", which, "state=", s.state, "fv_datasets_alive=", alive)
