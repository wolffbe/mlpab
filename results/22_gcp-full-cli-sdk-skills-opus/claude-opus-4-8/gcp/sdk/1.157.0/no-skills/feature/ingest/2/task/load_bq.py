import os
from google.cloud import bigquery

project = os.environ['GCP_PROJECT']
dataset = os.environ['GCP_BQ_DATASET']
loc = os.environ['GCP_LOCATION']
c = bigquery.Client(project=project)

staging = f'{project}.{dataset}._transactions_staging'
schema = [
    bigquery.SchemaField('row_id', 'STRING'),
    bigquery.SchemaField('account_id', 'STRING'),
    bigquery.SchemaField('event_time', 'INT64'),
    bigquery.SchemaField('amount', 'FLOAT64'),
    bigquery.SchemaField('category', 'STRING'),
]
c.query(f"DROP TABLE IF EXISTS `{staging}`", location=loc).result()
job_config = bigquery.LoadJobConfig(
    schema=schema, skip_leading_rows=1,
    source_format=bigquery.SourceFormat.CSV,
    write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
)
for f in ['data/transactions_export_1.csv', 'data/transactions_export_2.csv']:
    with open(f, 'rb') as fh:
        job = c.load_table_from_file(fh, staging, job_config=job_config, location=loc)
    job.result()
    print('loaded', f, '->', job.output_rows, 'rows')

st = c.get_table(staging)
print('staging total rows:', st.num_rows)

final = f'{project}.{dataset}.transactions8cf1c0'
q = f"""
CREATE OR REPLACE TABLE `{final}` AS
SELECT row_id, account_id, event_time, amount, category,
       TIMESTAMP_MILLIS(event_time) AS feature_timestamp
FROM `{staging}`
QUALIFY ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY event_time) = 1
"""
c.query(q, location=loc).result()
ft = c.get_table(final)
print('final table rows:', ft.num_rows)
print('final schema:', [(s.name, s.field_type) for s in ft.schema])
r = list(c.query(f"SELECT COUNT(*) n, COUNT(DISTINCT row_id) d FROM `{final}`", location=loc).result())[0]
print('n=', r.n, 'distinct row_id=', r.d)
c.query(f"DROP TABLE IF EXISTS `{staging}`", location=loc).result()
print('done')
