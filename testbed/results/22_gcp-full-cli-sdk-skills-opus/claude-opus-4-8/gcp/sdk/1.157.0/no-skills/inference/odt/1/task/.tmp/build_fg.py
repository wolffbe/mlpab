import os
import google.cloud.aiplatform as aiplatform
import vertexai
from vertexai.resources.preview import FeatureGroup, FeatureGroupBigQuerySource

proj = os.environ['GCP_PROJECT']; ds = os.environ['GCP_BQ_DATASET']; loc = os.environ['GCP_LOCATION']
aiplatform.init(project=proj, location=loc, api_transport="rest")
vertexai.init(project=proj, location=loc)

uri = f"bq://{proj}.{ds}.scored3ace95"
src = FeatureGroupBigQuerySource(uri=uri, entity_id_columns=["request_id"])

# Feature group == the feature table "scored3ace95". Record key: request_id.
try:
    fg = FeatureGroup("scored3ace95")
    print("existing fg:", fg.name)
except Exception:
    fg = FeatureGroup.create("scored3ace95", source=src,
                             labels={"version": "1"}, description="scored3ace95 v1")
    print("created fg:", fg.name)

existing = {f.name for f in fg.list_features()}
for feat in ["account_id", "distance_deg", "score"]:
    if feat in existing:
        print("feature exists:", feat)
    else:
        fg.create_feature(feat)
        print("created feature:", feat)

print("features now:", [f.name for f in fg.list_features()])
