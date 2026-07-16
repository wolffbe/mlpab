import os, time
import google.cloud.aiplatform as aiplatform
import google.cloud.bigquery as bq
from google.api_core.exceptions import ResourceExhausted, NotFound, AlreadyExists

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']; ds = os.environ['GCP_BQ_DATASET']
aiplatform.init(project=proj, location=loc, api_transport="rest")

FS_ID = "accountsed4daa_fs"
ET_ID = "accountsed4daa"

# Legacy ingest requires feature_time as a BQ TIMESTAMP; build a cast source
# table inside the dataset (deliverable table keeps updated_at as epoch-ms).
bqc = bq.Client(project=proj)
ingest_tbl = f"{proj}.{ds}.accountsed4daa_ingest"
bqc.query(f"""
CREATE OR REPLACE TABLE `{ingest_tbl}` AS
SELECT row_id, status, balance, TIMESTAMP_MILLIS(updated_at) AS feature_ts
FROM `{proj}.{ds}.accountsed4daa`
""").result()
print("ingest source table ready:", ingest_tbl)
bq_uri = f"bq://{ingest_tbl}"

# 1) Featurestore with online serving nodes (low-latency serving; separate quota)
fstore = None
try:
    fstore = aiplatform.Featurestore(FS_ID)
    print("featurestore exists:", fstore.resource_name)
except NotFound:
    pass

for i in range(6):
    if fstore is not None:
        break
    try:
        fstore = aiplatform.Featurestore.create(
            featurestore_id=FS_ID, online_store_fixed_node_count=1)
        print("featurestore created:", fstore.resource_name)
    except ResourceExhausted as e:
        print(f"fs attempt {i+1}: quota exhausted", str(e)[:70])
        if i < 5:
            time.sleep(40)

if fstore is None:
    print("LEGACY_ONLINE_BLOCKED: Featurestores quota exhausted")
    raise SystemExit(2)

# 2) EntityType (the feature table); entity id = record key row_id
try:
    et = fstore.get_entity_type(ET_ID)
    print("entity type exists:", et.resource_name)
except Exception:
    et = fstore.create_entity_type(entity_type_id=ET_ID,
                                   description="accounts v1; key row_id, event-time updated_at")
    print("entity type created:", et.resource_name)

# 3) Value features
existing = {f.name for f in et.list_features()}
specs = {"status": "STRING", "balance": "DOUBLE"}
for fid, vt in specs.items():
    if fid in existing:
        print("feature exists:", fid); continue
    et.create_feature(feature_id=fid, value_type=vt)
    print("feature created:", fid, vt)

# 4) Ingest from the deduped BQ table; feature_time=updated_at (event-time),
#    entity_id_field=row_id (record key). One row per entity -> latest revision.
et.ingest_from_bq(
    feature_ids=list(specs.keys()),
    feature_time="feature_ts",
    bq_source_uri=bq_uri,
    entity_id_field="row_id",
)
print("ingest complete")
print("DONE_LEGACY")
