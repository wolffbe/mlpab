import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as fs

proj = os.environ['GCP_PROJECT']; ds = os.environ['GCP_BQ_DATASET']
loc = os.environ['GCP_LOCATION']; prefix = os.environ['MLPAB_GCP_PREFIX']
aiplatform.init(project=proj, location=loc)

# --- verify registered Vertex model + metrics ---
vid = f"{prefix}_airqmodelf3f1d8"
models = aiplatform.Model.list(filter=f'display_name="{vid}"')
print("Vertex models matching:", [(m.display_name, m.resource_name) for m in models])

# --- offline feature group registry for airqf3f1d8 ---
try:
    fg = fs.FeatureGroup.create(
        name="airqf3f1d8",
        source=fs.FeatureGroupBigQuerySource(
            uri=f"bq://{proj}.{ds}.airqf3f1d8",
            entity_id_columns=["date"]),
    )
    print("FeatureGroup created:", fg.resource_name)
except Exception as e:
    print("FeatureGroup:", type(e).__name__, str(e)[:180])

# --- online store (low-latency) ---
store_name = f"{prefix}_airq_fos"
try:
    store = fs.FeatureOnlineStore.create_optimized_store(name=store_name)
    print("Online store created:", store.resource_name)
except Exception as e:
    print("online store create:", type(e).__name__, str(e)[:180])
    store = fs.FeatureOnlineStore(store_name)
    print("using existing:", store.resource_name)

# --- feature view over predictions table, entity key = date ---
try:
    fv = store.create_feature_view(
        name="airqpredf3f1d8",
        source=fs.FeatureViewBigQuerySource(
            uri=f"bq://{proj}.{ds}.airqpredf3f1d8",
            entity_id_columns=["date"]),
        sync_config="0 * * * *",
    )
    print("FeatureView created:", fv.resource_name)
except Exception as e:
    print("feature view create:", type(e).__name__, str(e)[:200])
    fv = fs.FeatureView(name="airqpredf3f1d8", feature_online_store_id=store_name)
    print("using existing:", fv.resource_name)

# --- trigger sync so predictions land in the online store ---
try:
    sync = fv.sync()
    print("sync triggered:", sync)
except Exception as e:
    print("sync:", type(e).__name__, str(e)[:200])
