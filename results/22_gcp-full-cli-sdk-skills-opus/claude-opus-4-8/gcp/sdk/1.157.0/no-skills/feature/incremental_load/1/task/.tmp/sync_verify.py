import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview.feature_store import FeatureView

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']
aiplatform.init(project=proj, location=loc, api_transport="rest")

FV_RES = ("projects/1014453977696/locations/***REDACTED***/"
          "featureOnlineStores/mlpaba45c1a_txn85a07a_store/featureViews/incrementaljob0872b7")
fv = FeatureView(FV_RES)

d = fv.to_dict()
print("=== FeatureView config ===")
print("name:", d.get("name"))
print("syncConfig:", d.get("syncConfig"))
print("bigQuerySource:", d.get("bigQuerySource"))

print("=== triggering immediate sync ===")
resp = fv.sync()
print("sync started:", getattr(resp, "resource_name", resp))
