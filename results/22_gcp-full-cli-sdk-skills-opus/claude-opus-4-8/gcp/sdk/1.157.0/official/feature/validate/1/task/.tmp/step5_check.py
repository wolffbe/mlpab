import os, json
from google.cloud import bigquery

bq = bigquery.Client(project=os.environ["GCP_PROJECT"], location=os.environ["GCP_LOCATION"])
t = f"{os.environ['GCP_PROJECT']}.{os.environ['GCP_BQ_DATASET']}.eventsd3c188"

r = list(bq.query(f"SELECT COUNT(*) c, MIN(amount) mn, MAX(amount) mx, COUNT(DISTINCT category) cats FROM `{t}`").result())[0]
print("feature_table_rows", r.c, "amount_min", r.mn, "amount_max", r.mx, "distinct_categories", r.cats)

bad = list(bq.query(
    f"SELECT COUNT(*) c FROM `{t}` WHERE amount IS NULL OR amount < 0 OR amount > 10000 "
    f"OR category NOT IN ('grocery','travel','salary','rent','other')").result())[0].c
print("violating_rows_in_table_should_be_0", bad)

a = json.load(open("submission/answers.json"))
print("rejected_count", len(a["rejected"]), "unique", len(set(a["rejected"])))
print("valid_plus_rejected", r.c + len(a["rejected"]))
