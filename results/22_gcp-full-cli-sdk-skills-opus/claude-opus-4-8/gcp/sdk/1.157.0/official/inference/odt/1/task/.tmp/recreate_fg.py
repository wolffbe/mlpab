import os
import time
import google.cloud.aiplatform as aiplatform
from google.api_core.exceptions import Conflict
from vertexai.resources.preview import feature_store as fs
from vertexai.resources.preview.feature_store import utils as fsutils

project = os.environ["GCP_PROJECT"]
dataset = os.environ["GCP_BQ_DATASET"]
loc = os.environ["GCP_LOCATION"]
aiplatform.init(project=project, location=loc, api_transport="rest")

TABLE = "scored3ace95"
table_uri = f"bq://{project}.{dataset}.{TABLE}"

fg = None
for attempt in range(20):
    try:
        fg = fs.FeatureGroup.create(
            name=TABLE,
            source=fsutils.FeatureGroupBigQuerySource(
                uri=table_uri, entity_id_columns=["request_id"]
            ),
            labels={"version": "1"},
            description="Scored requests feature table (version 1)",
        )
        print("recreated:", fg.resource_name)
        break
    except Conflict as e:
        msg = str(e)
        if "being deleted" in msg:
            print(f"attempt {attempt}: still deleting, waiting...")
            time.sleep(20)
            continue
        # already exists (a concurrent run recreated it) -> adopt if correct
        fg = fs.FeatureGroup(TABLE)
        cur = fg.gca_resource.big_query.big_query_source.input_uri
        print("exists with source:", cur)
        if cur == table_uri:
            break
        raise

if fg is None:
    raise SystemExit("could not (re)create FeatureGroup")

existing = {f.name for f in fg.list_features()}
for feat in ["account_id", "distance_deg", "score"]:
    if feat not in existing:
        fg.create_feature(name=feat)
        print("  feature created:", feat)

fg = fs.FeatureGroup(TABLE)
print("final source:", fg.gca_resource.big_query.big_query_source.input_uri)
print("final entity_id_columns:", list(fg.gca_resource.big_query.entity_id_columns))
print("final features:", sorted(f.name for f in fg.list_features()))
