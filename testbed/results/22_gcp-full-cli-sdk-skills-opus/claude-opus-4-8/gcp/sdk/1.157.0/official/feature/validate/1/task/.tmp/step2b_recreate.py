import os
import vertexai
from vertexai.resources import preview as fs
from vertexai.resources.preview.feature_store import utils

PROJECT = os.environ["GCP_PROJECT"]
DATASET = os.environ["GCP_BQ_DATASET"]
LOCATION = os.environ["GCP_LOCATION"]

vertexai.init(project=PROJECT, location=LOCATION, api_transport="rest")

BQ_URI = f"bq://{PROJECT}.{DATASET}.eventsd3c188"
FG_NAME = "eventsd3c188"

# Delete stale FeatureGroup (points at a different run's dataset), then recreate.
try:
    old = fs.FeatureGroup(FG_NAME)
    src = old.gca_resource.big_query.big_query_source.input_uri
    if src != BQ_URI:
        print("Deleting stale FeatureGroup pointing at:", src)
        old.delete(force=True)
        print("deleted")
    else:
        print("FeatureGroup already correct:", src)
except Exception as e:
    print("no existing FeatureGroup / lookup:", type(e).__name__)

# Recreate pointing at MY dataset
try:
    fg = fs.FeatureGroup(FG_NAME)
    print("exists after delete? source:", fg.gca_resource.big_query.big_query_source.input_uri)
except Exception:
    fg = fs.FeatureGroup.create(
        FG_NAME,
        source=utils.FeatureGroupBigQuerySource(uri=BQ_URI, entity_id_columns=["row_id"]),
        description="events feature table (contract-valid rows only)",
    )
    print("Created FeatureGroup:", fg.resource_name)

existing = {f.name for f in fg.list_features()}
for col in ["account_id", "event_time", "amount", "category"]:
    if col not in existing:
        fg.create_feature(col)
        print("created feature:", col)
print("source uri:", fg.gca_resource.big_query.big_query_source.input_uri)
print("entity cols:", list(fg.gca_resource.big_query.entity_id_columns))
print("features:", [f.name for f in fg.list_features()])
