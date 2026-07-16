import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as fs
from vertexai.resources.preview.feature_store import utils as fsutils

project = os.environ["GCP_PROJECT"]
dataset = os.environ["GCP_BQ_DATASET"]
loc = os.environ["GCP_LOCATION"]
aiplatform.init(project=project, location=loc, api_transport="rest")

TABLE = "scored3ace95"
table_uri = f"bq://{project}.{dataset}.{TABLE}"

# Existing FeatureGroup is a stale leftover pointing at another run's dataset.
fg = fs.FeatureGroup(TABLE)
cur = fg.gca_resource.big_query.big_query_source.input_uri
print("current source:", cur, "-> want:", table_uri)
if cur != table_uri:
    print("deleting stale FeatureGroup...")
    fg.delete(force=True)
    print("deleted; recreating against my dataset")
    fg = fs.FeatureGroup.create(
        name=TABLE,
        source=fsutils.FeatureGroupBigQuerySource(
            uri=table_uri, entity_id_columns=["request_id"]
        ),
        labels={"version": "1"},
        description="Scored requests feature table (version 1)",
    )
    print("recreated:", fg.resource_name)

existing = {f.name for f in fg.list_features()}
for feat in ["account_id", "distance_deg", "score"]:
    if feat not in existing:
        fg.create_feature(name=feat)
        print("  feature created:", feat)

fg = fs.FeatureGroup(TABLE)
print("final source:", fg.gca_resource.big_query.big_query_source.input_uri)
print("final entity_id_columns:", list(fg.gca_resource.big_query.entity_id_columns))
print("final features:", sorted(f.name for f in fg.list_features()))
