import os
import google.cloud.aiplatform as aiplatform
from google.cloud.aiplatform import Featurestore, EntityType
from google.api_core.exceptions import ResourceExhausted

PROJECT=os.environ["GCP_PROJECT"]; LOCATION=os.environ["GCP_LOCATION"]; DATASET=os.environ["GCP_BQ_DATASET"]
aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")

FS_ID = "accountsed4daa_fs"
ET_ID = "accountsed4daa"
BQ_URI = f"bq://{PROJECT}.{DATASET}.accountsed4daa"

# --- Featurestore with online serving enabled (1 fixed node) ---
try:
    store = Featurestore(FS_ID)
    print("Featurestore exists:", store.resource_name)
except Exception:
    try:
        store = Featurestore.create(FS_ID, online_store_fixed_node_count=1, labels={"version": "1"})
        print("Created Featurestore:", store.resource_name)
    except ResourceExhausted as e:
        print("QUOTA_EXHAUSTED_LEGACY:", e)
        raise SystemExit(3)

# --- EntityType keyed by row_id ---
try:
    et = store.get_entity_type(ET_ID)
    print("EntityType exists:", et.resource_name)
except Exception:
    et = store.create_entity_type(ET_ID, description="accounts feature table v1; key row_id, event-time updated_at")
    print("Created EntityType:", et.resource_name)

# --- Features ---
existing = {f.name for f in et.list_features()}
print("existing features:", existing)
for fid, vt in [("status", "STRING"), ("balance", "DOUBLE"), ("updated_at", "INT64")]:
    if fid not in existing:
        et.create_feature(fid, value_type=vt)
        print("created feature:", fid, vt)

# --- Ingest from BigQuery into online store; feature_timestamp is the event time ---
et.ingest_from_bq(
    feature_ids=["status", "balance", "updated_at"],
    feature_time="feature_timestamp",
    bq_source_uri=BQ_URI,
    entity_id_field="row_id",
)
print("INGEST_DONE")
print("STAGE_D_DONE")
