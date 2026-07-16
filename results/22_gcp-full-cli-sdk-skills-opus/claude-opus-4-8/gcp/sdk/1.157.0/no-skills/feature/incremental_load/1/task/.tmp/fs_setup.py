import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview.feature_store import (
    FeatureGroup, FeatureOnlineStore,
    FeatureGroupBigQuerySource, FeatureViewBigQuerySource,
)

proj = os.environ['GCP_PROJECT']
loc = os.environ['GCP_LOCATION']
ds = os.environ['GCP_BQ_DATASET']
prefix = os.environ['MLPAB_GCP_PREFIX']

aiplatform.init(project=proj, location=loc, api_transport="rest")

bq_uri = f"bq://{proj}.{ds}.incremental0872b7"
FG_NAME = "incremental0872b7"          # feature table (record key row_id, event-time event_time)
FOS_NAME = f"{prefix}_online_store"    # online (low-latency) store
FV_NAME = "incrementaljob0872b7"       # recurring scheduled ingestion job (daily cron)

# 1) Offline feature table registration: FeatureGroup over the BigQuery source.
try:
    fg = FeatureGroup(FG_NAME)
    print("FeatureGroup already exists:", fg.resource_name)
except Exception:
    fg = FeatureGroup.create(
        FG_NAME,
        source=FeatureGroupBigQuerySource(uri=bq_uri, entity_id_columns=["row_id"]),
        description="Events feature table; record key row_id, event-time event_time (epoch ms).",
    )
    print("Created FeatureGroup:", fg.resource_name)

# Register the feature columns on the feature table.
for col in ["account_id", "event_time", "amount", "category"]:
    try:
        f = fg.create_feature(col)
        print("  created feature:", col)
    except Exception as e:
        print("  feature", col, "->", type(e).__name__, str(e)[:120])

# 2) Online (low-latency) store.
try:
    fos = FeatureOnlineStore(FOS_NAME)
    print("Online store already exists:", fos.resource_name)
except Exception:
    fos = FeatureOnlineStore.create_optimized_store(FOS_NAME)
    print("Created online store:", fos.resource_name)

# 3) Recurring daily ingestion job = FeatureView with an attached cron sync schedule.
#    This serves features online AND syncs new increments from BigQuery every day.
existing = {fv.name: fv for fv in fos.list_feature_views()}
if FV_NAME in existing:
    fv = existing[FV_NAME]
    print("FeatureView already exists:", fv.resource_name)
else:
    fv = fos.create_feature_view(
        FV_NAME,
        source=FeatureViewBigQuerySource(uri=bq_uri, entity_id_columns=["row_id"]),
        sync_config="0 6 * * *",  # daily at 06:00 -> recurring ingestion schedule
    )
    print("Created FeatureView (daily cron):", fv.resource_name)

print("DONE")
print("FG_RESOURCE=", fg.resource_name)
print("FOS_RESOURCE=", fos.resource_name)
print("FV_RESOURCE=", fv.resource_name)
