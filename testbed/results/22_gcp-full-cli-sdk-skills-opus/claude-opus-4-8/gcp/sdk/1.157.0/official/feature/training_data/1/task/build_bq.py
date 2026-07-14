import os
from google.cloud import bigquery

PROJECT = os.environ["GCP_PROJECT"]
DATASET = os.environ["GCP_BQ_DATASET"]
client = bigquery.Client(project=PROJECT)
ds_ref = f"{PROJECT}.{DATASET}"

SF = bigquery.SchemaField
schemas = {
    "transactions": [SF("account_id", "STRING"), SF("event_time", "INT64"),
                     SF("amount", "FLOAT64"), SF("balance", "FLOAT64")],
    "transactions_late": [SF("account_id", "STRING"), SF("event_time", "INT64"),
                          SF("amount", "FLOAT64"), SF("balance", "FLOAT64")],
    "profiles": [SF("account_id", "STRING"), SF("event_time", "INT64"),
                 SF("credit_score", "INT64"), SF("tier", "STRING")],
    "activity": [SF("account_id", "STRING"), SF("event_time", "INT64"),
                 SF("sessions_7d", "INT64")],
    "account_health": [SF("account_id", "STRING"), SF("event_time", "INT64"),
                       SF("health_score", "FLOAT64")],
    "labels": [SF("account_id", "STRING"), SF("label_time", "INT64"),
               SF("churned", "INT64")],
}

for name, schema in schemas.items():
    table_id = f"{ds_ref}.{name}"
    job_config = bigquery.LoadJobConfig(
        schema=schema, source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1, write_disposition="WRITE_TRUNCATE",
    )
    with open(f"data/{name}.csv", "rb") as fh:
        job = client.load_table_from_file(fh, table_id, job_config=job_config)
    job.result()
    tbl = client.get_table(table_id)
    print(f"loaded {name}: {tbl.num_rows} rows")

print("ALL LOADED")
