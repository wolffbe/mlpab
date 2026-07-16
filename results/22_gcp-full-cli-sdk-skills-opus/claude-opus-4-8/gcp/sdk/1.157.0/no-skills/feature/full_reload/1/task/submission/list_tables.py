import os
from google.cloud import bigquery

bq = bigquery.Client(project=os.environ["GCP_PROJECT"])
ds = os.environ["GCP_BQ_DATASET"]
proj = os.environ["GCP_PROJECT"]
for t in bq.list_tables(ds):
    tb = bq.get_table(f"{proj}.{ds}.{t.table_id}")
    cols = [f.name for f in tb.schema]
    print(t.table_id, tb.num_rows, cols)
