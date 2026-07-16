import os
import vertexai
from vertexai.resources import preview as fs
from vertexai.resources.preview.feature_store import utils

PROJECT = os.environ["GCP_PROJECT"]
DATASET = os.environ["GCP_BQ_DATASET"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]

vertexai.init(project=PROJECT, location=LOCATION, api_transport="rest")

BQ_URI = f"bq://{PROJECT}.{DATASET}.eventsd3c188"
FG_NAME = "eventsd3c188"

# ---- Offline: FeatureGroup registered on the BigQuery feature table ----
try:
    fg = fs.FeatureGroup(FG_NAME)
    print("FeatureGroup already exists:", fg.resource_name)
except Exception:
    fg = fs.FeatureGroup.create(
        FG_NAME,
        source=utils.FeatureGroupBigQuerySource(
            uri=BQ_URI,
            entity_id_columns=["row_id"],
        ),
        description="events feature table (contract-valid rows only)",
    )
    print("Created FeatureGroup:", fg.resource_name)

# Register feature columns (row_id is the entity/record key)
existing = {f.name for f in fg.list_features()}
for col in ["account_id", "event_time", "amount", "category"]:
    if col in existing:
        print("feature exists:", col)
        continue
    fg.create_feature(col)
    print("created feature:", col)

print("features:", [f.name for f in fg.list_features()])
