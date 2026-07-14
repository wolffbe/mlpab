import os, time
import google.cloud.aiplatform as aiplatform
import vertexai
from vertexai.resources.preview import feature_store as fs
from google.api_core.exceptions import Conflict

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']; ds = os.environ['GCP_BQ_DATASET']
aiplatform.init(project=proj, location=loc, api_transport="rest")
vertexai.init(project=proj, location=loc)
want_uri = f"bq://{proj}.{ds}.accountsed4daa"

fg = None
for i in range(20):
    try:
        fg = fs.FeatureGroup.create(
            name="accountsed4daa",
            source=fs.FeatureGroupBigQuerySource(uri=want_uri, entity_id_columns=["row_id"]),
            description="accounts feature table v1; record key row_id, event-time updated_at (epoch ms)",
        )
        break
    except Conflict as e:
        print(f"attempt {i+1}: name still deleting, waiting...")
        time.sleep(15)

if fg is None:
    raise SystemExit("could not recreate FeatureGroup")

print("recreated FeatureGroup:", fg.resource_name)
for col in ["status", "balance", "updated_at"]:
    fg.create_feature(name=col)
    print("  feature:", col)
print("source uri:", fg.source.uri)
print("entity_id_columns:", list(fg.source.entity_id_columns))
print("features:", [f.name for f in fg.list_features()])
print("RECREATE_DONE")
