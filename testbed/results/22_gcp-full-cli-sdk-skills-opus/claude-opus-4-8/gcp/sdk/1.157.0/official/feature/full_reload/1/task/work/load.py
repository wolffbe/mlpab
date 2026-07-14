import os
import google.cloud.bigquery as bq

project = os.environ['GCP_PROJECT']
dataset = os.environ['GCP_BQ_DATASET']
client = bq.Client(project=project)


def load(table_id, csv_path, schema):
    ref = f"{project}.{dataset}.{table_id}"
    job_config = bq.LoadJobConfig(
        schema=schema,
        skip_leading_rows=1,
        source_format=bq.SourceFormat.CSV,
        write_disposition=bq.WriteDisposition.WRITE_TRUNCATE,
    )
    with open(csv_path, "rb") as f:
        job = client.load_table_from_file(f, ref, job_config=job_config)
    job.result()
    t = client.get_table(ref)
    print(f"{table_id}: {t.num_rows} rows, cols={[s.name for s in t.schema]}")


# Version 1 (initial export)
load("customerscd1186_1", "data/initial_export.csv", [
    bq.SchemaField("row_id", "STRING"),
    bq.SchemaField("name", "STRING"),
    bq.SchemaField("balance_eur", "FLOAT64"),
    bq.SchemaField("updated_at", "INT64"),
])

# Version 2 (new export)
load("customerscd1186_2", "data/reload/new_export.csv", [
    bq.SchemaField("row_id", "STRING"),
    bq.SchemaField("full_name", "STRING"),
    bq.SchemaField("balance", "FLOAT64"),
    bq.SchemaField("currency", "STRING"),
    bq.SchemaField("updated_at", "INT64"),
])
