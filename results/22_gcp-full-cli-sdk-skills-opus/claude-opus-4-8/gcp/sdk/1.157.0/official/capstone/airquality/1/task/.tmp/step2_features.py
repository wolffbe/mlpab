import os
from google.cloud import bigquery

proj = os.environ['GCP_PROJECT']; ds = os.environ['GCP_BQ_DATASET']
c = bigquery.Client(project=proj); D = f"`{proj}.{ds}`"

# Feature group: weather + lag + rolling air-quality signals + seasonal.
# Rolling features use ONLY past days (ROWS BETWEEN ... PRECEDING) => point-in-time correct.
fg_sql = f"""
CREATE OR REPLACE TABLE {D}.airqf3f1d8 AS
SELECT
  date,
  pm25_lag1,
  temperature, humidity, wind_speed, pressure, precipitation,
  EXTRACT(MONTH FROM date) AS month,
  EXTRACT(DAYOFYEAR FROM date) AS doy,
  AVG(pm25_lag1) OVER w3 AS pm25_lag1_roll3,
  AVG(pm25_lag1) OVER w7 AS pm25_lag1_roll7,
  AVG(temperature) OVER w3 AS temp_roll3,
  AVG(precipitation) OVER w3 AS precip_roll3,
  pm25
FROM {D}.airq_hist_raw
WINDOW
  w3 AS (ORDER BY date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING),
  w7 AS (ORDER BY date ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING)
"""
c.query(fg_sql).result()
t = c.get_table(f"{proj}.{ds}.airqf3f1d8")
print("airqf3f1d8 rows:", t.num_rows, "cols:", [f.name for f in t.schema])

# correlation of pm25_lag1 to target (sanity)
r = list(c.query(f"SELECT CORR(pm25_lag1,pm25) c, STDDEV(pm25) s FROM {D}.airq_hist_raw").result())[0]
print("corr(pm25_lag1,pm25)=", round(r.c,4), " stddev(pm25)=", round(r.s,4))
