import os
import google.cloud.aiplatform_v1 as v1
import google.cloud.bigquery as bigquery
from google.api_core.exceptions import NotFound

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']
parent = "projects/{}/locations/{}".format(proj, loc)
api_endpoint = "{}-aiplatform.googleapis.com".format(loc)
adm = v1.FeatureOnlineStoreAdminServiceClient(client_options={"api_endpoint": api_endpoint}, transport="rest")
reg = v1.FeatureRegistryServiceClient(client_options={"api_endpoint": api_endpoint}, transport="rest")
bq = bigquery.Client(project=proj)

def dataset_exists(uri):
    # uri like bq://proj.dataset.table
    try:
        body = uri.replace("bq://", "")
        p, dset, tbl = body.split(".", 2)
        bq.get_dataset("{}.{}".format(p, dset))
        return True
    except Exception:
        return False

stores = list(adm.list_feature_online_stores(parent=parent))
print("TOTAL online stores:", len(stores))
for s in stores:
    typ = "bigtable" if s.bigtable else ("optimized" if s.optimized else "?")
    fvs = list(adm.list_feature_views(parent=s.name))
    print("STORE", s.name.split('/')[-1], "state=", s.state, "type=", typ, "num_fv=", len(fvs))
    for fv in fvs:
        src = ""
        if fv.big_query_source and fv.big_query_source.uri:
            src = fv.big_query_source.uri
        elif fv.feature_registry_source:
            fgids = [g.feature_group_id for g in fv.feature_registry_source.feature_groups]
            src = "registry:" + ",".join(fgids)
        print("     FV", fv.name.split('/')[-1], "src=", src)
