import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as fs
aiplatform.init(project=os.environ["GCP_PROJECT"], location=os.environ["GCP_LOCATION"], api_transport="rest")
stores = fs.FeatureOnlineStore.list()
print("num stores:", len(stores))
for s in stores:
    try:
        fvs = s.list_feature_views()
    except Exception as e:
        fvs = f"err {e}"
    print("-", s.name, "| type", s.feature_online_store_type, "| created", s.create_time,
          "| labels", dict(s.labels), "| views", [v.name for v in fvs] if not isinstance(fvs,str) else fvs)
