import os
import google.cloud.aiplatform as aiplatform
from google.api_core.exceptions import ResourceExhausted
from vertexai.resources.preview.feature_store import (
    FeatureGroup, FeatureOnlineStore, FeatureViewBigQuerySource,
)

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']
ds = os.environ['GCP_BQ_DATASET']; prefix = os.environ['MLPAB_GCP_PREFIX']
aiplatform.init(project=proj, location=loc, api_transport="rest")

bq_uri = f"bq://{proj}.{ds}.incremental0872b7"
FG_NAME = "incremental0872b7"
FOS_NAME = f"{prefix}_online_store"
FV_NAME = "incrementaljob0872b7"

fg = FeatureGroup(FG_NAME)
print("FeatureGroup:", fg.resource_name)

# Register feature columns (version_column_name maps to the BQ column).
existing_feats = set()
try:
    existing_feats = {f.name for f in fg.list_features()}
except Exception as e:
    print("list_features:", e)
for col in ["account_id", "event_time", "amount", "category"]:
    if col in existing_feats:
        print("  feature exists:", col); continue
    try:
        fg.create_feature(col, version_column_name=col)
        print("  created feature:", col)
    except Exception as e:
        print("  feature", col, "->", type(e).__name__, str(e)[:160])

# Online store: try to create mine; fall back to an existing optimized store.
fos = None
try:
    fos = FeatureOnlineStore(FOS_NAME)
    print("Online store already exists:", fos.resource_name)
except Exception:
    try:
        fos = FeatureOnlineStore.create_optimized_store(FOS_NAME)
        print("Created online store:", fos.resource_name)
    except ResourceExhausted as e:
        print("QUOTA on create_optimized_store:", str(e)[:160])
        for s in FeatureOnlineStore.list():
            if s.feature_online_store_type.name == "OPTIMIZED":
                fos = s
                print("Reusing existing optimized online store:", fos.resource_name)
                break
if fos is None:
    raise SystemExit("No online store available (quota exhausted, none reusable).")

# Recurring daily ingestion job = FeatureView with attached cron sync schedule.
existing_fv = {fv.name: fv for fv in fos.list_feature_views()}
if FV_NAME in existing_fv:
    fv = existing_fv[FV_NAME]
    print("FeatureView already exists:", fv.resource_name)
else:
    fv = fos.create_feature_view(
        FV_NAME,
        source=FeatureViewBigQuerySource(uri=bq_uri, entity_id_columns=["row_id"]),
        sync_config="0 6 * * *",
    )
    print("Created FeatureView (daily cron):", fv.resource_name)

print("DONE")
print("FV_RESOURCE=", fv.resource_name)
