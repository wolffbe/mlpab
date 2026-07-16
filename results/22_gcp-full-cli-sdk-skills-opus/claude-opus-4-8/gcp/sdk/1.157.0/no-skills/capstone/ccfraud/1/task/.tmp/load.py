from google.cloud import bigquery
import os
proj=os.environ['GCP_PROJECT']; ds=os.environ['GCP_BQ_DATASET']
c=bigquery.Client(project=proj)

schema_common=[
 bigquery.SchemaField("transaction_id","STRING"),
 bigquery.SchemaField("cc_num","STRING"),
 bigquery.SchemaField("datetime","TIMESTAMP"),
 bigquery.SchemaField("amount","FLOAT"),
 bigquery.SchemaField("merchant","STRING"),
 bigquery.SchemaField("category","STRING"),
 bigquery.SchemaField("lat","FLOAT"),
 bigquery.SchemaField("long","FLOAT"),
]
schema_tx=schema_common+[bigquery.SchemaField("is_fraud","INTEGER")]

def load(path, table, schema):
    jc=bigquery.LoadJobConfig(schema=schema, skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV, write_disposition="WRITE_TRUNCATE")
    with open(path,"rb") as f:
        job=c.load_table_from_file(f, f"{proj}.{ds}.{table}", job_config=jc)
    job.result()
    t=c.get_table(f"{proj}.{ds}.{table}")
    print(table, t.num_rows, "rows")

load("data/transactions.csv","stg_transactions",schema_tx)
load("data/score_transactions.csv","stg_score",schema_common)
print("done")
