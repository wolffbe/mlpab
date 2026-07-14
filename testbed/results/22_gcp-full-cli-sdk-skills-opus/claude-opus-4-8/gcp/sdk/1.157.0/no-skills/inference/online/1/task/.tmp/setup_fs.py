import os
import google.cloud.aiplatform as aiplatform
import vertexai
from vertexai.resources.preview import feature_store as fss

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]

aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")
vertexai.init(project=PROJECT, location=LOCATION)

FG_ID = "profilesaf22bf"
BQ_URI = f"bq://{PROJECT}.{DATASET}.{FG_ID}"
STORE_ID = f"{PREFIX}_profilesaf22bf_store"
FV_ID = "profilesaf22bf"
FEATURES = ["f1", "f2", "f3", "f4"]

# 1. FeatureGroup (the "feature table") with record key account_id
try:
    fg = fss.FeatureGroup(FG_ID)
    print("FeatureGroup exists:", fg.resource_name)
except Exception:
    fg = fss.FeatureGroup.create(
        FG_ID,
        source=fss.FeatureGroupBigQuerySource(uri=BQ_URI, entity_id_columns=["account_id"]),
    )
    print("Created FeatureGroup:", fg.resource_name)

# 2. Register features f1..f4
existing = set()
try:
    existing = {f.name for f in fg.list_features()}
except Exception as e:
    print("list_features err", e)
for feat in FEATURES:
    if feat in existing:
        print("feature exists:", feat); continue
    fg.create_feature(feat, version_column_name=feat, description=f"feature {feat}")
    print("created feature:", feat)

# 3. Optimized online store (online / low-latency access)
try:
    store = fss.FeatureOnlineStore(STORE_ID)
    print("OnlineStore exists:", store.resource_name)
except Exception:
    store = fss.FeatureOnlineStore.create_optimized_store(STORE_ID)
    print("Created OnlineStore:", store.resource_name)

# 4. FeatureView from the registry (feature group) referencing f1..f4
feat_refs = [f"{FG_ID}.{f}" for f in FEATURES]
try:
    fv = fss.FeatureView(FV_ID, feature_online_store_id=STORE_ID)
    print("FeatureView exists:", fv.resource_name)
except Exception:
    fv = store.create_feature_view(
        FV_ID,
        source=fss.FeatureViewRegistrySource(features=feat_refs),
    )
    print("Created FeatureView:", fv.resource_name)

print("DONE_SETUP", store.resource_name, fv.resource_name)
