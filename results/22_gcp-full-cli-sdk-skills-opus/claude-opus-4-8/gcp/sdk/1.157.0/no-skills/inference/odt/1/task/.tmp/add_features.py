import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import FeatureGroup

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']
aiplatform.init(project=proj, location=loc, api_transport="rest")

fg = FeatureGroup("scored3ace95")
existing = {f.name for f in fg.list_features()}
for feat in ["account_id", "distance_deg", "score"]:
    if feat in existing:
        print("exists:", feat); continue
    fg.create_feature(feat, description=f"{feat} feature")
    print("created:", feat)
print("features:", [f.name for f in fg.list_features()])
