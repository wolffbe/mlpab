import os
import google.cloud.bigquery as bigquery

proj = os.environ['GCP_PROJECT']
ds = os.environ['GCP_BQ_DATASET']
c = bigquery.Client(project=proj)
d = c.get_dataset("{}.{}".format(proj, ds))
print("dataset location:", d.location)
print("existing tables:", [t.table_id for t in c.list_tables(d)])
