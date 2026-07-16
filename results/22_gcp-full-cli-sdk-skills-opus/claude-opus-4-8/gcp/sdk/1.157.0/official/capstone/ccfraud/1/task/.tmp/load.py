import os
from google.cloud import bigquery

proj = os.environ['GCP_PROJECT']; ds = os.environ['GCP_BQ_DATASET']
bq = bigquery.Client(project=proj)

def load(csv_path, table, schema):
    tid = f"{proj}.{ds}.{table}"
    job_config = bigquery.LoadJobConfig(
        schema=schema, skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
        write_disposition="WRITE_TRUNCATE",
    )
    with open(csv_path, "rb") as f:
        job = bq.load_table_from_file(f, tid, job_config=job_config)
    job.result()
    t = bq.get_table(tid)
    print(f"loaded {table}: {t.num_rows} rows")

txn_schema = [
    bigquery.SchemaField("transaction_id","STRING"),
    bigquery.SchemaField("cc_num","STRING"),
    bigquery.SchemaField("datetime","TIMESTAMP"),
    bigquery.SchemaField("amount","FLOAT"),
    bigquery.SchemaField("merchant","STRING"),
    bigquery.SchemaField("category","STRING"),
    bigquery.SchemaField("lat","FLOAT"),
    bigquery.SchemaField("long","FLOAT"),
    bigquery.SchemaField("is_fraud","INTEGER"),
]
score_schema = txn_schema[:-1]

load("data/transactions.csv","raw_txn",txn_schema)
load("data/score_transactions.csv","raw_score",score_schema)
print("done")
