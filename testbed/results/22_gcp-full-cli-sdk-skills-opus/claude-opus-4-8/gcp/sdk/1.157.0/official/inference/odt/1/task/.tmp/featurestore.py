import os
import google.cloud.aiplatform as aiplatform
from google.cloud import bigquery
from vertexai.resources.preview import feature_store as fs
from vertexai.resources.preview.feature_store import utils as fsutils

project = os.environ["GCP_PROJECT"]
dataset = os.environ["GCP_BQ_DATASET"]
loc = os.environ["GCP_LOCATION"]
prefix = os.environ["MLPAB_GCP_PREFIX"]

aiplatform.init(project=project, location=loc, api_transport="rest")
bq = bigquery.Client(project=project, location=loc)
ds = f"{project}.{dataset}"

TABLE = "scored3ace95"
table_uri = f"bq://{project}.{dataset}.{TABLE}"
view_name = f"{TABLE}_online"
view_uri = f"bq://{project}.{dataset}.{view_name}"

# Online serving source needs a feature_timestamp column; keep it out of the
# canonical feature table by deriving a view.
bq.query(
    f"""CREATE OR REPLACE VIEW `{ds}.{view_name}` AS
        SELECT request_id, account_id, distance_deg, score,
               CURRENT_TIMESTAMP() AS feature_timestamp
        FROM `{ds}.{TABLE}`""",
    location=loc,
).result()
print("view ready:", view_name)


def get_or(create_fn, get_fn):
    try:
        return get_fn()
    except Exception:
        return create_fn()


# ---- Offline feature table: FeatureGroup (record key = request_id) ----
try:
    fg = fs.FeatureGroup(TABLE)
    print("FeatureGroup exists:", fg.resource_name)
except Exception:
    fg = fs.FeatureGroup.create(
        name=TABLE,
        source=fsutils.FeatureGroupBigQuerySource(
            uri=table_uri, entity_id_columns=["request_id"]
        ),
        labels={"version": "1"},
        description="Scored requests feature table (version 1)",
    )
    print("FeatureGroup created:", fg.resource_name)

existing_feats = {f.name for f in fg.list_features()}
print("existing features:", existing_feats)
for feat in ["account_id", "distance_deg", "score"]:
    if feat in existing_feats:
        print("  feature exists:", feat)
        continue
    f = fg.create_feature(name=feat)
    print("  feature created:", f.name)

# ---- Online serving: FeatureOnlineStore + FeatureView ----
store_name = f"{prefix}_scored3ace95_store"
try:
    online_store = fs.FeatureOnlineStore(store_name)
    print("FeatureOnlineStore exists:", online_store.resource_name)
except Exception:
    online_store = fs.FeatureOnlineStore.create_optimized_store(
        name=store_name, labels={"version": "1"}
    )
    print("FeatureOnlineStore created:", online_store.resource_name)

existing_views = {v.name for v in online_store.list_feature_views()}
print("existing views:", existing_views)
if TABLE in existing_views:
    fv = fs.FeatureView(TABLE, feature_online_store_id=store_name)
    print("FeatureView exists:", fv.resource_name)
else:
    fv = online_store.create_feature_view(
        name=TABLE,
        source=fsutils.FeatureViewBigQuerySource(
            uri=view_uri, entity_id_columns=["request_id"]
        ),
        labels={"version": "1"},
    )
    print("FeatureView created:", fv.resource_name)

# Trigger online sync so features are available for low-latency lookup.
sync = fv.sync()
print("sync started:", sync)
