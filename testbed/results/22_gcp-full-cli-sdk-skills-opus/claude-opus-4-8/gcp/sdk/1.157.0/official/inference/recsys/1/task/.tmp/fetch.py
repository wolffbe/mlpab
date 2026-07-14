import os
import google.cloud.aiplatform_v1 as v1

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']
parent = "projects/{}/locations/{}".format(proj, loc)
api_endpoint = "{}-aiplatform.googleapis.com".format(loc)
OS_ID = "recs75dfad_os"; FV_ID = "recs75dfad"
fv_name = "{}/featureOnlineStores/{}/featureViews/{}".format(parent, OS_ID, FV_ID)

data = v1.FeatureOnlineStoreServiceClient(client_options={"api_endpoint": api_endpoint}, transport="rest")

for key in ["U0000#1", "U0001#3", "U0039#5"]:
    req = v1.FetchFeatureValuesRequest(
        feature_view=fv_name,
        data_key=v1.FeatureViewDataKey(key=key),
    )
    try:
        resp = data.fetch_feature_values(request=req)
        pairs = [(f.name, str(f.value)) for f in resp.key_values.features] if resp.key_values else resp
        print(key, "->", pairs)
    except Exception as e:
        print(key, "ERROR", repr(e))
