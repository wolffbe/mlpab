import os
import google.cloud.bigquery as bq

project = os.environ['GCP_PROJECT']
dataset = os.environ['GCP_BQ_DATASET']
client = bq.Client(project=project)

for table_id in ("customerscd1186_1", "customerscd1186_2"):
    ref = f"`{project}.{dataset}.{table_id}`"
    # Add the platform-required feature_timestamp derived from the event-time
    # column updated_at (epoch milliseconds). Data/feature columns unchanged.
    sql = (
        f"CREATE OR REPLACE TABLE {ref} AS "
        f"SELECT *, TIMESTAMP_MILLIS(updated_at) AS feature_timestamp FROM {ref}"
    )
    client.query(sql).result()
    t = client.get_table(f"{project}.{dataset}.{table_id}")
    print(f"{table_id}: {t.num_rows} rows, cols={[s.name for s in t.schema]}")
