import os, time
import vertexai
from vertexai.resources import preview as fs
from vertexai.resources.preview.feature_store import utils
from google.api_core.exceptions import Conflict

PROJECT = os.environ["GCP_PROJECT"]
DATASET = os.environ["GCP_BQ_DATASET"]
LOCATION = os.environ["GCP_LOCATION"]

vertexai.init(project=PROJECT, location=LOCATION, api_transport="rest")
BQ_URI = f"bq://{PROJECT}.{DATASET}.eventsd3c188"
FG_NAME = "eventsd3c188"

fg = None
for attempt in range(40):
    try:
        fg = fs.FeatureGroup.create(
            FG_NAME,
            source=utils.FeatureGroupBigQuerySource(uri=BQ_URI, entity_id_columns=["row_id"]),
            description="events feature table (contract-valid rows only)",
        )
        print("Created FeatureGroup:", fg.resource_name)
        break
    except Conflict as e:
        print(f"attempt {attempt}: still deleting, waiting...")
        time.sleep(30)
if fg is None:
    raise SystemExit("FeatureGroup still not creatable after retries")

for col in ["account_id", "event_time", "amount", "category"]:
    fg.create_feature(col)
    print("created feature:", col)

print("source uri:", fg.gca_resource.big_query.big_query_source.input_uri)
print("entity cols:", list(fg.gca_resource.big_query.entity_id_columns))
print("features:", [f.name for f in fg.list_features()])
