import os
import google.cloud.aiplatform as aiplatform
import vertexai
from vertexai.resources.preview import feature_store as fs

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']; ds = os.environ['GCP_BQ_DATASET']
aiplatform.init(project=proj, location=loc, api_transport="rest")
vertexai.init(project=proj, location=loc)

bq_uri = f"bq://{proj}.{ds}.accountsed4daa"

# 1) FeatureGroup (offline registration)
try:
    fg = fs.FeatureGroup("accountsed4daa")
    print("FeatureGroup exists:", fg.resource_name)
except Exception:
    fg = fs.FeatureGroup.create(
        name="accountsed4daa",
        source=fs.FeatureGroupBigQuerySource(uri=bq_uri, entity_id_columns=["row_id"]),
        description="accounts feature table v1; record key row_id, event-time updated_at",
    )
    print("FeatureGroup created:", fg.resource_name)

existing = {f.name for f in fg.list_features()}
print("existing features:", existing)
for col in ["status", "balance", "updated_at"]:
    if col in existing:
        print("  feature exists:", col); continue
    f = fg.create_feature(name=col)
    print("  feature created:", f.resource_name)

print("DONE_FG")
