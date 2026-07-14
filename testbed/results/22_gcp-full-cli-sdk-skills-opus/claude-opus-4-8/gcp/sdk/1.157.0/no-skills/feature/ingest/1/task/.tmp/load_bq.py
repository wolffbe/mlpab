import os
from google.cloud import bigquery

project = os.environ["GCP_PROJECT"]
dataset = os.environ["GCP_BQ_DATASET"]
client = bigquery.Client(project=project)

staging = f"{project}.{dataset}.transactions85a07a_staging"
final = f"{project}.{dataset}.transactions85a07a"

schema = [
    bigquery.SchemaField("row_id", "STRING"),
    bigquery.SchemaField("account_id", "STRING"),
    bigquery.SchemaField("event_time", "INT64"),
    bigquery.SchemaField("amount", "FLOAT64"),
    bigquery.SchemaField("category", "STRING"),
]

client.query(f"DROP TABLE IF EXISTS `{staging}`").result()

for f in ["data/transactions_export_1.csv", "data/transactions_export_2.csv"]:
    cfg = bigquery.LoadJobConfig(
        schema=schema,
        skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    with open(f, "rb") as fh:
        job = client.load_table_from_file(fh, staging, job_config=cfg)
    job.result()
    print("loaded", f, "->", job.output_rows, "rows")

print("staging total rows:", client.get_table(staging).num_rows)

# Dedup by row_id on-platform; add feature_timestamp (TIMESTAMP) derived from
# event_time (epoch ms) as required by Vertex Feature Store feature groups,
# while preserving event_time (bigint epoch ms) as the documented event-time.
dedup_sql = f"""
CREATE OR REPLACE TABLE `{final}` AS
SELECT row_id, account_id, event_time, amount, category,
       TIMESTAMP_MILLIS(event_time) AS feature_timestamp
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY event_time) AS rn
  FROM `{staging}`
)
WHERE rn = 1
"""
client.query(dedup_sql).result()

t = client.get_table(final)
print("final table:", final)
print("final rows:", t.num_rows)
print("final schema:", [(f.name, f.field_type) for f in t.schema])

# sanity: distinct row_ids
r = list(client.query(
    f"SELECT COUNT(*) AS n, COUNT(DISTINCT row_id) AS d, "
    f"MIN(row_id) AS mn, MAX(row_id) AS mx FROM `{final}`"
).result())[0]
print("count=%d distinct=%d min=%s max=%s" % (r.n, r.d, r.mn, r.mx))

client.query(f"DROP TABLE IF EXISTS `{staging}`").result()
print("dropped staging")
