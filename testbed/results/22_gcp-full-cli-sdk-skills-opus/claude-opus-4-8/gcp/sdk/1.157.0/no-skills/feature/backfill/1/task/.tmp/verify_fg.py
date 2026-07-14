import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as fs
PROJECT=os.environ["GCP_PROJECT"]; LOCATION=os.environ["GCP_LOCATION"]; DATASET=os.environ["GCP_BQ_DATASET"]
aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")
fg = fs.FeatureGroup("accountsed4daa")
print("FeatureGroup:", fg.resource_name)
print("  source uri:", fg.source.uri if fg.source else None)
print("  entity_id_columns:", getattr(fg.source, "entity_id_columns", None))
print("  labels:", dict(fg.labels))
print("  features:", [f.name for f in fg.list_features()])
