import os
from google.cloud import aiplatform_v1 as a
proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']; prefix = os.environ['MLPAB_GCP_PREFIX']
ep = f"{loc}-aiplatform.googleapis.com"; copts = {"api_endpoint": ep}
parent = f"projects/{proj}/locations/{loc}"

admin = a.FeatureOnlineStoreAdminServiceClient(transport="rest", client_options=copts)
print("== FeatureOnlineStores ==")
for s in admin.list_feature_online_stores(parent=parent):
    print(" ", s.name.split('/')[-1], s.state.name if hasattr(s,'state') else '')

fss = a.FeaturestoreServiceClient(transport="rest", client_options=copts)
print("== Legacy Featurestores ==")
for f in fss.list_featurestores(parent=parent):
    print(" ", f.name.split('/')[-1], f.state.name)
