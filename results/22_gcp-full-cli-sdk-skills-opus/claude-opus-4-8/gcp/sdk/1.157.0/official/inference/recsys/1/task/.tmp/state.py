import os
import google.cloud.aiplatform_v1 as v1
from google.api_core.exceptions import NotFound

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']
parent = "projects/{}/locations/{}".format(proj, loc)
api_endpoint = "{}-aiplatform.googleapis.com".format(loc)
reg = v1.FeatureRegistryServiceClient(client_options={"api_endpoint": api_endpoint}, transport="rest")
adm = v1.FeatureOnlineStoreAdminServiceClient(client_options={"api_endpoint": api_endpoint}, transport="rest")

print("=== FeatureGroups ===")
for fg in reg.list_feature_groups(parent=parent):
    bq = fg.big_query.big_query_source.input_uri if fg.big_query else None
    print(fg.name, "| bq=", bq)
print("=== FeatureOnlineStores ===")
for s in adm.list_feature_online_stores(parent=parent):
    print(s.name, "| state=", s.state)
    for fv in adm.list_feature_views(parent=s.name):
        print("   FV:", fv.name)
