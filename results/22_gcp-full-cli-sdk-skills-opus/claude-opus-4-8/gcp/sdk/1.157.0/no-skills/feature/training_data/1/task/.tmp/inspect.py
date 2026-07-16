import os
from google.cloud import bigquery

client = bigquery.Client(project=os.environ['GCP_PROJECT'])
ds = os.environ['GCP_BQ_DATASET']
dsref = f"{os.environ['GCP_PROJECT']}.{ds}"
print("dataset:", dsref)
d = client.get_dataset(dsref)
print("location:", d.location)
tables = list(client.list_tables(dsref))
print("existing tables:", [t.table_id for t in tables])
