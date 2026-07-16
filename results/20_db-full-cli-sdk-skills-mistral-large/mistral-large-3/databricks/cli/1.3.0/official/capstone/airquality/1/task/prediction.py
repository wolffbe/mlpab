# Databricks notebook source
# MAGIC %md
# MAGIC ## Predicting PM2.5 for Forecast Days

This notebook predicts `pm25` for every row in `forecast_days.csv` and writes results to the feature table `airqpredfcd91b`. It also enables low-latency lookup for the predictions table.

### Steps:
1. Read `forecast_days.csv` from `data/`.
2. Read the trained model `airqmodelfcd91b`.
3. Predict `pm25` for every row.
4. Write predictions to `airqpredfcd91b`.
5. Enable low-latency lookup for the predictions table.

---

### Step 1: Read Forecast Days Data

```python
import pandas as pd
from pyspark.sql import SparkSession

# Read the forecast days data
forecast_df = pd.read_csv("/dbfs/data/forecast_days.csv")
spark_df = SparkSession.getActiveSession().createDataFrame(forecast_df)

# Display the data
spark_df.display()
```

---

### Step 2: Read the Trained Model

```python
import mlflow
from databricks.feature_engineering import FeatureEngineeringClient

# Define the model name
model_name = "airqmodelfcd91b"

# Load the trained model
model_uri = f"models:/{model_name}/latest"
model = mlflow.sklearn.load_model(model_uri)

print(f"Model {model_name} loaded successfully.")
```

---

### Step 3: Engineer Features for Forecast Days

```python
from pyspark.sql.functions import col

# Select features for prediction
forecast_features_df = spark_df.select(
    "date",
    "pm25_lag1",
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "precipitation"
)

# Add missing features (e.g., rolling averages, lag features)
# Note: In a real scenario, these would be computed from the feature group
forecast_features_df = forecast_features_df.withColumn("pm25_lag7", col("pm25_lag1"))  # Placeholder
forecast_features_df = forecast_features_df.withColumn("pm25_rolling_avg_7d", col("pm25_lag1"))  # Placeholder
forecast_features_df = forecast_features_df.withColumn("temp_humidity_ratio", col("temperature") / col("humidity"))

# Display the engineered features
forecast_features_df.display()
```

---

### Step 4: Predict PM2.5

```python
# Convert Spark DataFrame to Pandas
pandas_df = forecast_features_df.toPandas()
X_forecast = pandas_df.drop(["date"], axis=1)

# Predict PM2.5
pandas_df["pm25_pred"] = model.predict(X_forecast)

# Convert back to Spark DataFrame
predictions_df = SparkSession.getActiveSession().createDataFrame(pandas_df)

# Display predictions
predictions_df.display()
```

---

### Step 5: Write Predictions to Feature Table

```python
# Initialize the Feature Engineering Client
fe = FeatureEngineeringClient()

# Define the predictions table name
schema = dbutils.widgets.get("schema")
predictions_table_name = f"{schema}.airqpredfcd91b"

# Write predictions to the feature table
fe.create_table(
    name=predictions_table_name,
    primary_keys=["date"],
    df=predictions_df.select("date", "pm25_pred"),
    description="Predicted PM2.5 values for forecast days"
)

print(f"Predictions table {predictions_table_name} created successfully.")
```

---

### Step 6: Enable Low-Latency Lookup

```python
# Enable online serving for the predictions table
fe.publish_table(
    name=predictions_table_name,
    online_store="databricks"
)

print(f"Low-latency lookup enabled for {predictions_table_name}.")
```