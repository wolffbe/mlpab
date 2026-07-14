import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as fs

project = os.environ["GCP_PROJECT"]
loc = os.environ["GCP_LOCATION"]
aiplatform.init(project=project, location=loc, api_transport="rest")

TABLE = "scored3ace95"
fg = fs.FeatureGroup(TABLE)
existing = {f.name for f in fg.list_features()}
print("existing:", existing)
for feat in ["account_id", "distance_deg", "score"]:
    if feat in existing:
        print("exists:", feat)
        continue
    f = fg.create_feature(name=feat, version_column_name=feat)
    print("created:", f.name)

fg = fs.FeatureGroup(TABLE)
print("features now:", sorted(f.name for f in fg.list_features()))
