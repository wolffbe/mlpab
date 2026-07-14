import os, glob
from google.cloud import bigquery

proj = os.environ['GCP_PROJECT']; ds = os.environ['GCP_BQ_DATASET']
c = bigquery.Client(project=proj)
table_id = f"{proj}.{ds}.incremental0872b7"

schema = [
    bigquery.SchemaField("row_id", "STRING"),
    bigquery.SchemaField("account_id", "STRING"),
    bigquery.SchemaField("event_time", "INT64"),
    bigquery.SchemaField("amount", "FLOAT64"),
    bigquery.SchemaField("category", "STRING"),
]
c.query(f"DROP TABLE IF EXISTS `{table_id}`").result()

job_config = bigquery.LoadJobConfig(
    schema=schema,
    skip_leading_rows=1,
    source_format=bigquery.SourceFormat.CSV,
    write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
)
for f in sorted(glob.glob("data/increment_*.csv")):
    with open(f, "rb") as fh:
        job = c.load_table_from_file(fh, table_id, job_config=job_config)
    job.result()
    print("loaded", f, "->", job.output_rows, "rows")

# add feature_timestamp (event-time) column derived from epoch millis
c.query(
    f"CREATE OR REPLACE TABLE `{table_id}` AS "
    f"SELECT *, TIMESTAMP_MILLIS(event_time) AS feature_timestamp FROM `{table_id}`"
).result()

t = c.get_table(table_id)
print("final rows:", t.num_rows)
print("schema:", [(fld.name, fld.field_type) for fld in t.schema])
