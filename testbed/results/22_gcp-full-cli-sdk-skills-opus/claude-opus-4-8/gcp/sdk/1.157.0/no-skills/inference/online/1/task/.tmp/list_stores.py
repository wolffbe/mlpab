import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as fss

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")

print("=== FeatureOnlineStores ===")
for s in fss.FeatureOnlineStore.list():
    fvs = []
    try:
        fvs = [fv.name for fv in s.list_feature_views()]
    except Exception as e:
        fvs = [f"ERR {e}"]
    print(s.name, "| type:", s.feature_online_store_type, "| mine:", s.name.startswith(PREFIX), "| views:", fvs)
