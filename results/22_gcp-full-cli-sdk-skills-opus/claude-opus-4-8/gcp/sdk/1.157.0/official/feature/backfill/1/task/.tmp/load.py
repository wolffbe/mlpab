import google.cloud.bigquery as bq
import os
proj = os.environ['GCP_PROJECT']; ds = os.environ['GCP_BQ_DATASET']
client = bq.Client(project=proj)
staging = f"{proj}.{ds}.accountsed4daa_staging"

schema = [
    bq.SchemaField("row_id", "STRING"),
    bq.SchemaField("status", "STRING"),
    bq.SchemaField("balance", "FLOAT64"),
    bq.SchemaField("updated_at", "INT64"),
]
for i, csv in enumerate(["data/batch_1.csv", "data/batch_2.csv", "data/batch_3.csv"]):
    disp = bq.WriteDisposition.WRITE_TRUNCATE if i == 0 else bq.WriteDisposition.WRITE_APPEND
    cfg = bq.LoadJobConfig(schema=schema, skip_leading_rows=1,
                           source_format=bq.SourceFormat.CSV, write_disposition=disp)
    with open(csv, "rb") as f:
        job = client.load_table_from_file(f, staging, job_config=cfg)
    job.result()
    print("loaded", csv, "->", job.output_rows, "rows")

print("staging total rows:", client.get_table(staging).num_rows)

final = f"{proj}.{ds}.accountsed4daa"
sql = f"""
CREATE OR REPLACE TABLE `{final}` AS
SELECT row_id, status, balance, updated_at
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY updated_at DESC) rn
  FROM `{staging}`
)
WHERE rn = 1
"""
client.query(sql).result()
print("final table rows:", client.get_table(final).num_rows)
d = list(client.query(f"SELECT COUNT(DISTINCT row_id) c FROM `{staging}`").result())[0].c
print("distinct row_ids in staging:", d)
