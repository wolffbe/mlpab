import os
from google.cloud import bigquery

bq = bigquery.Client(project=os.environ["GCP_PROJECT"])
proj = os.environ["GCP_PROJECT"]
ds = os.environ["GCP_BQ_DATASET"]

for base in ["customerscd1186_1", "customerscd1186_2"]:
    src = f"{proj}.{ds}.{base}_fg"
    sql = (
        f"CREATE OR REPLACE TABLE `{src}` AS "
        f"SELECT *, TIMESTAMP_MILLIS(updated_at) AS feature_timestamp "
        f"FROM `{proj}.{ds}.{base}`"
    )
    bq.query(sql).result()
    tb = bq.get_table(src)
    print(f"{src}: {tb.num_rows} rows, cols={[f.name for f in tb.schema]}")
