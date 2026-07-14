import os
from google.cloud import bigquery

proj = os.environ['GCP_PROJECT']; ds = os.environ['GCP_BQ_DATASET']
c = bigquery.Client(project=proj); D = f"`{proj}.{ds}`"

pred_sql = f"""
CREATE OR REPLACE TABLE {D}.airqpredf3f1d8 AS
SELECT
  CAST(date AS STRING) AS date,
  CAST(predicted_pm25 AS FLOAT64) AS pm25_pred
FROM ML.PREDICT(MODEL {D}.airqmodelf3f1d8,
  (SELECT
     date, pm25_lag1, temperature, humidity, wind_speed, pressure, precipitation,
     EXTRACT(MONTH FROM date) AS month,
     EXTRACT(DAYOFYEAR FROM date) AS doy
   FROM {D}.airq_forecast_raw))
ORDER BY date
"""
c.query(pred_sql).result()
rows = list(c.query(f"SELECT COUNT(*) n, MIN(pm25_pred) mn, MAX(pm25_pred) mx, AVG(pm25_pred) av FROM {D}.airqpredf3f1d8").result())[0]
print("airqpredf3f1d8:", rows.n, "rows  min=%.3f max=%.3f avg=%.3f" % (rows.mn, rows.mx, rows.av))
for r in list(c.query(f"SELECT date, pm25_pred FROM {D}.airqpredf3f1d8 ORDER BY date LIMIT 5").result()):
    print(" ", r.date, round(r.pm25_pred,3))
print("schema:", [f.name+':'+f.field_type for f in c.get_table(f'{proj}.{ds}.airqpredf3f1d8').schema])
