import os
import time
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as fs
from google.api_core.exceptions import Conflict, AlreadyExists

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]
aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")

fg_id = "customerscd1186_1"
src_full = f"{PROJECT}.{DATASET}.customerscd1186_1_fg"
feature_cols = ["name", "balance_eur", "updated_at"]

try:
    fg = fs.FeatureGroup(fg_id)
    print(f"FeatureGroup {fg_id} already exists")
except Exception:
    src = fs.FeatureGroupBigQuerySource(uri=f"bq://{src_full}", entity_id_columns=["row_id"])
    fg = None
    for attempt in range(12):
        try:
            fg = fs.FeatureGroup.create(
                name=fg_id, source=src,
                description="customers feature table v1; key=row_id, event-time=updated_at(epoch ms)",
                labels={"record_key": "row_id", "event_time": "updated_at"},
            )
            break
        except Conflict as e:
            print(f"  conflict (tombstone), retry {attempt}: {str(e)[:70]}")
            time.sleep(20)
    if fg is None:
        raise RuntimeError("v1 FeatureGroup name still tombstoned")
    print(f"created FeatureGroup {fg.name}")

existing = set()
try:
    existing = {f.name for f in fg.list_features()}
except Exception:
    pass
for col in feature_cols:
    if col in existing:
        print(f"  feature {col} already registered")
        continue
    try:
        fg.create_feature(name=col, version_column_name=col)
        print(f"  registered feature {col}")
    except AlreadyExists:
        print(f"  feature {col} already exists")
print("STEP3 DONE")
