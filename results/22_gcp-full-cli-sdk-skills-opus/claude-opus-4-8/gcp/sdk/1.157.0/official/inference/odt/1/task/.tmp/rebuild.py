import os
import time
import google.cloud.aiplatform as aiplatform
from google.cloud import bigquery
from google.api_core.exceptions import Conflict
from vertexai.resources.preview import feature_store as fs
from vertexai.resources.preview.feature_store import utils as fsutils

project = os.environ["GCP_PROJECT"]
dataset = os.environ["GCP_BQ_DATASET"]
loc = os.environ["GCP_LOCATION"]
aiplatform.init(project=project, location=loc, api_transport="rest")
bq = bigquery.Client(project=project, location=loc)
ds = f"{project}.{dataset}"

TABLE = "scored3ace95"
SRC = "scored3ace95_src"  # backing table with required feature_timestamp
src_uri = f"bq://{project}.{dataset}.{SRC}"

# Backing table = the 4 feature-table columns plus the platform-required
# feature_timestamp column. The deliverable table `scored3ace95` stays 4-col.
bq.query(
    f"""CREATE OR REPLACE TABLE `{ds}.{SRC}` AS
        SELECT request_id, account_id, distance_deg, score,
               CURRENT_TIMESTAMP() AS feature_timestamp
        FROM `{ds}.{TABLE}`""",
    location=loc,
).result()
print("backing table ready:", SRC)

fg = None
for attempt in range(30):
    try:
        fg = fs.FeatureGroup.create(
            name=TABLE,
            source=fsutils.FeatureGroupBigQuerySource(
                uri=src_uri, entity_id_columns=["request_id"]
            ),
            labels={"version": "1"},
            description="Scored requests feature table (version 1)",
        )
        print("FeatureGroup created:", fg.resource_name)
        break
    except Conflict as e:
        if "being deleted" in str(e):
            print(f"attempt {attempt}: name still releasing, waiting...")
            time.sleep(20)
            continue
        fg = fs.FeatureGroup(TABLE)
        print("adopting existing FeatureGroup:",
              fg.gca_resource.big_query.big_query_source.input_uri)
        break

if fg is None:
    raise SystemExit("could not create FeatureGroup")

existing = {f.name for f in fg.list_features()}
for feat in ["account_id", "distance_deg", "score"]:
    if feat not in existing:
        fg.create_feature(name=feat)
        print("  feature created:", feat)

fg = fs.FeatureGroup(TABLE)
print("final source:", fg.gca_resource.big_query.big_query_source.input_uri)
print("final entity_id_columns:", list(fg.gca_resource.big_query.entity_id_columns))
print("final features:", sorted(f.name for f in fg.list_features()))
