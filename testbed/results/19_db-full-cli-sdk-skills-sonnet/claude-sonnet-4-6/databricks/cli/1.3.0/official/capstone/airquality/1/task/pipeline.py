# Databricks notebook source

# COMMAND ----------
catalog = "workspace"
schema = "mlpab375647"
full_schema = f"{catalog}.{schema}"
fg_name = "airqfdfb59"
td_name = "airqtdfdfb59"
model_name = "airqmodelfdfb59"
pred_name = "airqpredfdfb59"
user = "benedict@logicalclocks.com"
prefix = "mlpab375647"
base_path = f"/Workspace/Users/{user}/{prefix}"

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, DateType
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# COMMAND ----------
# Load raw data
history_df = spark.read.csv(f"{base_path}/airquality_history.csv", header=True, inferSchema=True)
forecast_df = spark.read.csv(f"{base_path}/forecast_days.csv", header=True, inferSchema=True)

print(f"History rows: {history_df.count()}")
print(f"Forecast rows: {forecast_df.count()}")
history_df.printSchema()

# COMMAND ----------
# Feature engineering on history
w3 = Window.orderBy("date").rowsBetween(-2, 0)
w7 = Window.orderBy("date").rowsBetween(-6, 0)

history_feat = (history_df
    .withColumn("pm25_roll3", F.avg("pm25_lag1").over(w3))
    .withColumn("pm25_roll7", F.avg("pm25_lag1").over(w7))
    .withColumn("temp_roll3", F.avg("temperature").over(w3))
    .withColumn("wind_roll3", F.avg("wind_speed").over(w3))
    .withColumn("humidity_roll3", F.avg("humidity").over(w3))
    .withColumn("month", F.month("date").cast(DoubleType()))
    .withColumn("temp_x_humidity", F.col("temperature") * F.col("humidity"))
)

# Feature engineering on forecast
# Combine last 7 rows of history with forecast rows for rolling computation
history_tail = (history_df.orderBy(F.desc("date")).limit(7)
    .select("date","pm25_lag1","temperature","humidity","wind_speed","pressure","precipitation"))
forecast_with_hist = history_tail.withColumn("pm25", F.lit(None).cast(DoubleType())).union(
    forecast_df.withColumn("pm25", F.lit(None).cast(DoubleType()))
).orderBy("date")

forecast_feat = (forecast_with_hist
    .withColumn("pm25_roll3", F.avg("pm25_lag1").over(w3))
    .withColumn("pm25_roll7", F.avg("pm25_lag1").over(w7))
    .withColumn("temp_roll3", F.avg("temperature").over(w3))
    .withColumn("wind_roll3", F.avg("wind_speed").over(w3))
    .withColumn("humidity_roll3", F.avg("humidity").over(w3))
    .withColumn("month", F.month("date").cast(DoubleType()))
    .withColumn("temp_x_humidity", F.col("temperature") * F.col("humidity"))
)

# Keep only forecast dates
forecast_dates_list = [r["date"] for r in forecast_df.select("date").collect()]
forecast_feat_final = forecast_feat.filter(F.col("date").isin(forecast_dates_list))

print(f"History features: {history_feat.count()}")
print(f"Forecast features: {forecast_feat_final.count()}")

# COMMAND ----------
# Write feature group (history with engineered features, minus pm25 target)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {full_schema}")

fg_cols = ["date","pm25_lag1","temperature","humidity","wind_speed","pressure","precipitation",
           "pm25_roll3","pm25_roll7","temp_roll3","wind_roll3","humidity_roll3","month","temp_x_humidity","pm25"]

history_feat.select(fg_cols).write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(f"{full_schema}.{fg_name}")
print(f"Feature group written: {full_schema}.{fg_name}")

# COMMAND ----------
# Create training dataset with train/test split
feat_table = spark.table(f"{full_schema}.{fg_name}").dropna()
total = feat_table.count()
cutoff_idx = int(total * 0.8)
dates_ordered = [r["date"] for r in feat_table.orderBy("date").select("date").collect()]
cutoff_date = dates_ordered[cutoff_idx]

train_td = feat_table.filter(F.col("date") < F.lit(cutoff_date)).withColumn("split", F.lit("train"))
test_td  = feat_table.filter(F.col("date") >= F.lit(cutoff_date)).withColumn("split", F.lit("test"))
td_df = train_td.union(test_td)

td_df.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(f"{full_schema}.{td_name}")
print(f"Training dataset written: {full_schema}.{td_name}, train={train_td.count()}, test={test_td.count()}")

# COMMAND ----------
# Train model
feature_cols = ["pm25_lag1","temperature","humidity","wind_speed","pressure","precipitation",
                "pm25_roll3","pm25_roll7","temp_roll3","wind_roll3","humidity_roll3","month","temp_x_humidity"]
label_col = "pm25"

train_pd = train_td.toPandas()
test_pd  = test_td.toPandas()

X_train = train_pd[feature_cols].values
y_train = train_pd[label_col].values
X_test  = test_pd[feature_cols].values
y_test  = test_pd[label_col].values

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{user}/{prefix}/airq_exp")

with mlflow.start_run() as run:
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=3, random_state=42
    )
    model.fit(X_train, y_train)

    train_rmse = float(np.sqrt(mean_squared_error(y_train, model.predict(X_train))))
    test_rmse  = float(np.sqrt(mean_squared_error(y_test,  model.predict(X_test))))

    mlflow.log_params({"n_estimators":300,"max_depth":4,"learning_rate":0.05,"subsample":0.8})
    mlflow.log_metrics({"train_rmse": train_rmse, "test_rmse": test_rmse})
    print(f"Train RMSE={train_rmse:.4f}  Test RMSE={test_rmse:.4f}")

    signature = infer_signature(X_train, model.predict(X_train))
    model_info = mlflow.sklearn.log_model(
        model, "model",
        signature=signature,
        input_example=X_train[:5],
        registered_model_name=f"{full_schema}.{model_name}"
    )
    print(f"Model registered: {full_schema}.{model_name}")

# COMMAND ----------
# Predict forecast days
forecast_pd = forecast_feat_final.toPandas()
forecast_pd["pm25_pred"] = model.predict(forecast_pd[feature_cols].values)

pred_spark = spark.createDataFrame(forecast_pd[["date","pm25_pred"]])
pred_spark.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(f"{full_schema}.{pred_name}")
print(f"Predictions written: {full_schema}.{pred_name}")
spark.table(f"{full_schema}.{pred_name}").show(100)

# COMMAND ----------
# Create online table for low-latency lookup
import json, requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
ws_url = spark.conf.get("spark.databricks.workspaceUrl")
api_url = f"https://{ws_url}/api/2.0/online-tables"

ot_spec = {
    "name": f"{full_schema}.{pred_name}_online",
    "spec": {
        "source_table_full_name": f"{full_schema}.{pred_name}",
        "primary_key_columns": [{"name": "date"}],
        "run_triggered": {}
    }
}

resp = requests.post(api_url, headers={"Authorization": f"Bearer {token}"}, json=ot_spec)
print(f"Online table create status: {resp.status_code}")
print(resp.text[:500])
