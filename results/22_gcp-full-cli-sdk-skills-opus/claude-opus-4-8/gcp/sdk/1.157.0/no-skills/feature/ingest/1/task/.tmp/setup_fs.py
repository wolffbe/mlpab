import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import (
    FeatureGroup,
    FeatureGroupBigQuerySource,
    FeatureOnlineStore,
    FeatureViewRegistrySource,
)

project = os.environ["GCP_PROJECT"]
location = os.environ["GCP_LOCATION"]
dataset = os.environ["GCP_BQ_DATASET"]
prefix = os.environ["MLPAB_GCP_PREFIX"]

aiplatform.init(project=project, location=location, api_transport="rest")

FG_NAME = "transactions85a07a"          # the required feature table name, version 1
BQ_URI = f"bq://{project}.{dataset}.{FG_NAME}"
STORE_NAME = f"{prefix}_txn85a07a_store"  # prefixed for run isolation
VIEW_NAME = "transactions85a07a"          # view mirrors the feature table name
FEATURE_COLS = ["account_id", "event_time", "amount", "category"]

# 1) FeatureGroup on the BigQuery offline table (record key = row_id).
existing = {fg.name: fg for fg in FeatureGroup.list()}
if FG_NAME in existing:
    fg = existing[FG_NAME]
    print("FeatureGroup already exists:", fg.resource_name)
else:
    fg = FeatureGroup.create(
        FG_NAME,
        source=FeatureGroupBigQuerySource(uri=BQ_URI, entity_id_columns=["row_id"]),
        description="Transactions feature table v1 (record key row_id, event_time epoch ms)",
    )
    print("Created FeatureGroup:", fg.resource_name)

# 2) Register features.
have = {f.name for f in fg.list_features()}
for col in FEATURE_COLS:
    if col in have:
        print("feature exists:", col)
        continue
    fg.create_feature(col, version_column_name=col, description=f"feature {col}")
    print("created feature:", col)

# 3) Online store (optimized -> low-latency serving).
stores = {s.name: s for s in FeatureOnlineStore.list()}
if STORE_NAME in stores:
    store = stores[STORE_NAME]
    print("Online store exists:", store.resource_name)
else:
    store = FeatureOnlineStore.create_optimized_store(STORE_NAME)
    print("Created online store:", store.resource_name)

# 4) FeatureView referencing the registry features, with periodic sync.
views = {v.name: v for v in store.list_feature_views()}
if VIEW_NAME in views:
    view = views[VIEW_NAME]
    print("FeatureView exists:", view.resource_name)
else:
    view = store.create_feature_view(
        VIEW_NAME,
        source=FeatureViewRegistrySource(
            features=[f"{FG_NAME}.{c}" for c in FEATURE_COLS],
        ),
        sync_config="0 * * * *",
    )
    print("Created FeatureView:", view.resource_name)

print("VIEW_RESOURCE=" + view.resource_name)
print("STORE_RESOURCE=" + store.resource_name)
print("FG_RESOURCE=" + fg.resource_name)
