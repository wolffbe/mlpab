import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview.feature_store import FeatureGroup, FeatureOnlineStore

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']
aiplatform.init(project=proj, location=loc, api_transport="rest")

print("=== existing FeatureOnlineStores ===")
for s in FeatureOnlineStore.list():
    t = None
    try:
        t = s.feature_online_store_type
    except Exception as e:
        t = "?"
    print(" ", s.name, "| type=", t)

print("=== try create_feature with full error ===")
fg = FeatureGroup("incremental0872b7")
try:
    f = fg.create_feature("account_id")
    print("created", f.resource_name)
except Exception as e:
    print("FULL ERROR:", repr(e))
