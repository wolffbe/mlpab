import os
from google.cloud import bigquery

project = os.environ["GCP_PROJECT"]
dataset = os.environ["GCP_BQ_DATASET"]
client = bigquery.Client(project=project)

table_id = f"{project}.{dataset}.training_data"
schema = [bigquery.SchemaField("row_id", "STRING")] + \
    [bigquery.SchemaField(f"f{i}", "FLOAT64") for i in range(1, 7)] + \
    [bigquery.SchemaField("label", "INT64")]
job_config = bigquery.LoadJobConfig(
    schema=schema, skip_leading_rows=1,
    source_format=bigquery.SourceFormat.CSV,
    write_disposition="WRITE_TRUNCATE")
with open("data/training_data.csv", "rb") as f:
    job = client.load_table_from_file(f, table_id, job_config=job_config)
job.result()
tbl = client.get_table(table_id)
print("Loaded rows:", tbl.num_rows, "into", table_id)
