import os, time
import google.cloud.aiplatform as aiplatform
import google.cloud.bigquery as bq
import vertexai
from vertexai.resources.preview import feature_store as fs
from google.api_core.exceptions import Conflict

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']; ds = os.environ['GCP_BQ_DATASET']
aiplatform.init(project=proj, location=loc, api_transport="rest")
vertexai.init(project=proj, location=loc)

final = f"{proj}.{ds}.accountsed4daa"
# Add feature_timestamp (TIMESTAMP) as the new-FS event-time column, derived from
# the event-time updated_at (epoch ms). updated_at is kept per the table schema.
bqc = bq.Client(project=proj)
bqc.query(f"""
CREATE OR REPLACE TABLE `{final}` AS
SELECT row_id, status, balance, updated_at,
       TIMESTAMP_MILLIS(updated_at) AS feature_timestamp
FROM `{final}`
""").result()
print("added feature_timestamp column; rows:", bqc.get_table(final).num_rows)

want_uri = f"bq://{final}"
fg = None
for i in range(30):
    try:
        fg = fs.FeatureGroup.create(
            name="accountsed4daa",
            source=fs.FeatureGroupBigQuerySource(uri=want_uri, entity_id_columns=["row_id"]),
            description="accounts feature table v1; record key row_id, event-time updated_at (epoch ms)",
        )
        break
    except Conflict:
        print(f"attempt {i+1}: name still deleting, waiting...")
        time.sleep(15)

if fg is None:
    raise SystemExit("could not recreate FeatureGroup (name still reserved)")

print("FeatureGroup:", fg.resource_name)
for col in ["status", "balance", "updated_at"]:
    fg.create_feature(name=col)
print("source uri:", fg.source.uri)
print("entity_id_columns:", list(fg.source.entity_id_columns))
print("features:", [f.name for f in fg.list_features()])
print("FINALIZE_DONE")
