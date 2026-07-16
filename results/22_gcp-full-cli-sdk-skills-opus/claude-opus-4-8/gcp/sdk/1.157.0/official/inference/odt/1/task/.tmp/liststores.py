import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as fs

project = os.environ["GCP_PROJECT"]
loc = os.environ["GCP_LOCATION"]
prefix = os.environ["MLPAB_GCP_PREFIX"]
aiplatform.init(project=project, location=loc, api_transport="rest")

stores = fs.FeatureOnlineStore.list()
print("total stores:", len(stores))
for s in stores:
    try:
        views = s.list_feature_views()
        vnames = [v.name for v in views]
    except Exception as e:
        vnames = f"err {e}"
    mine = s.name.startswith(prefix)
    print(f"  {s.name}  mine={mine}  views={vnames}")
