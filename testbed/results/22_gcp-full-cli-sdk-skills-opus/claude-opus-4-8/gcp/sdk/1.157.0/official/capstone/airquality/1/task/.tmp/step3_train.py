import os
from google.cloud import bigquery

proj = os.environ['GCP_PROJECT']; ds = os.environ['GCP_BQ_DATASET']
prefix = os.environ['MLPAB_GCP_PREFIX']
c = bigquery.Client(project=proj); D = f"`{proj}.{ds}`"

# Training dataset: only features available at serve time for forecast_days
# (per-row lag1 + weather + seasonal) plus target. Time-based holdout flag.
td_sql = f"""
CREATE OR REPLACE TABLE {D}.airqtdf3f1d8 AS
WITH ranked AS (
  SELECT *, PERCENT_RANK() OVER (ORDER BY date) AS pr
  FROM {D}.airqf3f1d8
  WHERE pm25 IS NOT NULL
)
SELECT
  date, pm25_lag1, temperature, humidity, wind_speed, pressure,
  precipitation, month, doy, pm25,
  (pr >= 0.8) AS is_eval          -- last 20% by date = held-out
FROM ranked
"""
c.query(td_sql).result()
t = c.get_table(f"{proj}.{ds}.airqtdf3f1d8")
print("airqtdf3f1d8 rows:", t.num_rows)

vertex_id = f"{prefix}_airqmodelf3f1d8"
model_sql = f"""
CREATE OR REPLACE MODEL {D}.airqmodelf3f1d8
OPTIONS(
  model_type='BOOSTED_TREE_REGRESSOR',
  input_label_cols=['pm25'],
  data_split_method='CUSTOM',
  data_split_col='is_eval',
  max_iterations=50,
  learn_rate=0.1,
  early_stop=TRUE,
  min_rel_progress=0.001,
  subsample=0.85,
  model_registry='vertex_ai',
  vertex_ai_model_id='{vertex_id}',
  vertex_ai_model_version_aliases=['default']
) AS
SELECT pm25_lag1, temperature, humidity, wind_speed, pressure,
       precipitation, month, doy, pm25, is_eval
FROM {D}.airqtdf3f1d8
"""
print("training model -> vertex id:", vertex_id)
c.query(model_sql).result()
print("model trained.")

ev = list(c.query(f"SELECT * FROM ML.EVALUATE(MODEL {D}.airqmodelf3f1d8)").result())[0]
import math
rmse = math.sqrt(ev.mean_squared_error)
print("HELD-OUT EVAL: mae=%.4f mse=%.4f RMSE=%.4f r2=%.4f" % (
    ev.mean_absolute_error, ev.mean_squared_error, rmse, ev.r2_score))
