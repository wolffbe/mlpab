from google.cloud import bigquery

c = bigquery.Client(project='***REDACTED***')
DS = '***REDACTED***.mlpab_mlpab5f324c'
raw = f'{DS}.eventsd3c188_raw'
schema = [
    bigquery.SchemaField('row_id', 'STRING'),
    bigquery.SchemaField('account_id', 'STRING'),
    bigquery.SchemaField('event_time', 'INT64'),
    bigquery.SchemaField('amount', 'FLOAT64'),
    bigquery.SchemaField('category', 'STRING'),
]
job_config = bigquery.LoadJobConfig(
    schema=schema, skip_leading_rows=1,
    source_format=bigquery.SourceFormat.CSV,
    write_disposition='WRITE_TRUNCATE', allow_quoted_newlines=True,
)
with open('data/events.csv', 'rb') as f:
    job = c.load_table_from_file(f, raw, job_config=job_config)
job.result()
t = c.get_table(raw)
print('loaded rows:', t.num_rows)
