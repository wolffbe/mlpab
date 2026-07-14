import os
from google.cloud import bigquery

proj = os.environ['GCP_PROJECT']
ds = os.environ['GCP_BQ_DATASET']
c = bigquery.Client(project=proj)
dsref = f"{proj}.{ds}"
print("dataset location:", c.get_dataset(dsref).location)

def load(path, table, schema):
    job = bigquery.LoadJobConfig(
        schema=schema, skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV, write_disposition="WRITE_TRUNCATE")
    with open(path, "rb") as f:
        j = c.load_table_from_file(f, f"{dsref}.{table}", job_config=job)
        j.result()
    print("loaded", table, c.get_table(f'{dsref}.{table}').num_rows, "rows")

S = bigquery.SchemaField
hist_schema = [S("date","DATE"),S("pm25_lag1","FLOAT"),S("temperature","FLOAT"),
    S("humidity","FLOAT"),S("wind_speed","FLOAT"),S("pressure","FLOAT"),
    S("precipitation","FLOAT"),S("pm25","FLOAT")]
fc_schema = hist_schema[:-1]
load("data/airquality_history.csv", "airq_hist_raw", hist_schema)
load("data/forecast_days.csv", "airq_forecast_raw", fc_schema)
