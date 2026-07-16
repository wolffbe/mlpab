import os
import google.cloud.aiplatform as aiplatform
import vertexai
from vertexai.resources.preview import feature_store as fs

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']
aiplatform.init(project=proj, location=loc, api_transport="rest")
vertexai.init(project=proj, location=loc)

fg = fs.FeatureGroup("accountsed4daa")
print("FeatureGroup:", fg.name)
print("  source uri:", fg.source.uri)
print("  entity_id_columns:", list(fg.source.entity_id_columns))
print("  features:", [f.name for f in fg.list_features()])
