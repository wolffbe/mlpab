import os, time
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import (
    FeatureOnlineStore, FeatureView, FeatureViewBigQuerySource,
)

proj = os.environ['GCP_PROJECT']; ds = os.environ['GCP_BQ_DATASET']; loc = os.environ['GCP_LOCATION']
prefix = os.environ['MLPAB_GCP_PREFIX']
aiplatform.init(project=proj, location=loc, api_transport="rest")

store_name = f"{prefix}_online_store"
# Online store (low-latency serving layer). Optimized store supports FeatureView online serving.
existing = {s.name: s for s in FeatureOnlineStore.list()}
if store_name in existing:
    store = existing[store_name]
    print("existing online store:", store.name)
else:
    store = FeatureOnlineStore.create_optimized_store(store_name)
    print("created online store:", store.name)

uri = f"bq://{proj}.{ds}.scored3ace95"
src = FeatureViewBigQuerySource(uri=uri, entity_id_columns=["request_id"])

fvs = {v.name: v for v in store.list_feature_views()}
if "scored3ace95" in fvs:
    fv = fvs["scored3ace95"]
    print("existing feature view:", fv.name)
else:
    fv = store.create_feature_view("scored3ace95", source=src,
                                   sync_config="0 0 * * *", labels={"version": "1"})
    print("created feature view:", fv.name)

# Trigger a sync so data lands in the online store for low-latency lookup.
sync = fv.sync()
print("sync started:", sync.resource_name if hasattr(sync, 'resource_name') else sync)
EOF_MARK = None
