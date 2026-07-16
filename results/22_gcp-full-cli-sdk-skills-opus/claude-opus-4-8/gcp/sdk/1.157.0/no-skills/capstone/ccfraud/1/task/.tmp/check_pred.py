from google.cloud import bigquery
import os
proj=os.environ['GCP_PROJECT']; ds=os.environ['GCP_BQ_DATASET']; loc=os.environ['GCP_LOCATION']
c=bigquery.Client(project=proj)
D=f"`{proj}.{ds}`"
t=c.get_table(f"{proj}.{ds}.ccpred76ccb2")
print("ccpred76ccb2 rows", t.num_rows, "schema", [f.name for f in t.schema])
r=list(c.query(f"SELECT MIN(fraud_probability) mn, MAX(fraud_probability) mx, AVG(fraud_probability) av, COUNTIF(fraud_probability IS NULL) n_null FROM {D}.ccpred76ccb2", location=loc).result())[0]
print("stats", r.mn, r.mx, r.av, "nulls", r.n_null)
for row in c.query(f"SELECT * FROM {D}.ccpred76ccb2 LIMIT 3", location=loc).result():
    print(dict(row))
