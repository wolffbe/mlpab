import os
import google.cloud.aiplatform as aiplatform
import vertexai
from vertexai.resources.preview import feature_store as fs

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']
prefix = os.environ['MLPAB_GCP_PREFIX']
aiplatform.init(project=proj, location=loc, api_transport="rest")
vertexai.init(project=proj, location=loc)

stores = fs.FeatureOnlineStore.list()
print("total online stores:", len(stores))
for s in stores:
    try:
        st = s.gca_resource.state
    except Exception:
        st = "?"
    print(" -", s.name, "| state:", st, "| views:", [v.name for v in s.list_feature_views()])
print("MY PREFIX:", prefix)
