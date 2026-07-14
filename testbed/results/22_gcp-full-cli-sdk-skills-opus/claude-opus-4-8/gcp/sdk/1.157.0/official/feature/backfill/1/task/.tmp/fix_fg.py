import os
import google.cloud.aiplatform as aiplatform
import vertexai
from vertexai.resources.preview import feature_store as fs

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']; ds = os.environ['GCP_BQ_DATASET']
aiplatform.init(project=proj, location=loc, api_transport="rest")
vertexai.init(project=proj, location=loc)

want_uri = f"bq://{proj}.{ds}.accountsed4daa"

fg = fs.FeatureGroup("accountsed4daa")
if fg.source.uri != want_uri:
    print("stale FeatureGroup source:", fg.source.uri, "-> deleting")
    # delete features then the group
    for f in fg.list_features():
        try:
            f.delete()
        except Exception as e:
            print("  feature delete warn:", str(e)[:60])
    fg.delete()
    print("deleted stale FeatureGroup")
    fg = None

if fg is None:
    fg = fs.FeatureGroup.create(
        name="accountsed4daa",
        source=fs.FeatureGroupBigQuerySource(uri=want_uri, entity_id_columns=["row_id"]),
        description="accounts feature table v1; record key row_id, event-time updated_at (epoch ms)",
    )
    print("recreated FeatureGroup:", fg.resource_name)
    for col in ["status", "balance", "updated_at"]:
        fg.create_feature(name=col)
        print("  feature:", col)

print("source uri:", fg.source.uri)
print("entity_id_columns:", list(fg.source.entity_id_columns))
print("features:", [f.name for f in fg.list_features()])
print("FIX_DONE")
