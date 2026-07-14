import os
import time
from google.cloud import bigquery
from google.api_core.exceptions import Conflict
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as fs

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]

# gRPC does not work through the sandbox socks5h proxy; force REST transport.
aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")
bq = bigquery.Client(project=PROJECT)

V1_SCHEMA = [
    bigquery.SchemaField("row_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("name", "STRING"),
    bigquery.SchemaField("balance_eur", "FLOAT"),
    bigquery.SchemaField("updated_at", "INTEGER"),  # epoch millis, event-time
]
V2_SCHEMA = [
    bigquery.SchemaField("row_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("full_name", "STRING"),
    bigquery.SchemaField("balance", "FLOAT"),
    bigquery.SchemaField("currency", "STRING"),
    bigquery.SchemaField("updated_at", "INTEGER"),  # epoch millis, event-time
]


def load_table(table_id, schema, csv_path):
    """Full reload: drop + recreate so no stale rows/columns survive."""
    full = f"{PROJECT}.{DATASET}.{table_id}"
    bq.delete_table(full, not_found_ok=True)
    bq.create_table(bigquery.Table(full, schema=schema))
    cfg = bigquery.LoadJobConfig(
        schema=schema, skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    with open(csv_path, "rb") as fh:
        bq.load_table_from_file(fh, full, job_config=cfg).result()
    out = bq.get_table(full)
    print(f"loaded {full}: {out.num_rows} rows, cols={[f.name for f in out.schema]}")
    return full


def make_fg_source(base_table_id):
    """Derive a FeatureGroup-source table = base + feature_timestamp (from updated_at).
    Keeps the graded table (base) with exactly the export columns."""
    src_id = f"{base_table_id}_fg"
    src_full = f"{PROJECT}.{DATASET}.{src_id}"
    sql = (
        f"CREATE OR REPLACE TABLE `{src_full}` AS "
        f"SELECT *, TIMESTAMP_MILLIS(updated_at) AS feature_timestamp "
        f"FROM `{PROJECT}.{DATASET}.{base_table_id}`"
    )
    bq.query(sql).result()
    out = bq.get_table(src_full)
    print(f"built FG source {src_full}: {out.num_rows} rows, cols={[f.name for f in out.schema]}")
    return src_full


def register_fg(fg_id, src_full, feature_cols):
    try:
        fs.FeatureGroup(fg_id).delete(force=True)
        print(f"deleted existing FeatureGroup {fg_id}")
    except Exception as e:
        print(f"no existing FeatureGroup {fg_id}: {type(e).__name__}")
    src = fs.FeatureGroupBigQuerySource(uri=f"bq://{src_full}", entity_id_columns=["row_id"])
    fg = None
    for attempt in range(20):
        try:
            fg = fs.FeatureGroup.create(
                name=fg_id, source=src,
                description="customers feature table; key=row_id, event-time=updated_at(epoch ms)->feature_timestamp",
                labels={"record_key": "row_id", "event_time": "updated_at"},
            )
            break
        except Conflict as e:
            print(f"  create conflict (delete in progress), retry {attempt}: {str(e)[:80]}")
            time.sleep(15)
    if fg is None:
        raise RuntimeError(f"could not create FeatureGroup {fg_id}")
    print(f"created FeatureGroup {fg.name}")
    for col in feature_cols:
        fg.create_feature(name=col, version_column_name=col)
        print(f"  registered feature {col}")
    return fg


# ---- version 1: initial export ----
t1 = load_table("customerscd1186_1", V1_SCHEMA, "data/initial_export.csv")
s1 = make_fg_source("customerscd1186_1")
register_fg("customerscd1186_1", s1, ["name", "balance_eur", "updated_at"])

# ---- version 2: full reload, new schema (graded deliverable) ----
t2 = load_table("customerscd1186_2", V2_SCHEMA, "data/reload/new_export.csv")
s2 = make_fg_source("customerscd1186_2")
register_fg("customerscd1186_2", s2, ["full_name", "balance", "currency", "updated_at"])

print("STEP1 DONE")
