# Databricks notebook source
# MAGIC %md
# MAGIC ## Training a PM2.5 Regressor

This notebook assembles the training dataset `airqtdfcd91b`, trains a regressor `airqmodelfcd91b`, logs its metrics, and registers it.

### Steps:
1. Read features from the feature group `airqfcd91b`.
2. Assemble the training dataset `airqtdfcd91b`.
3. Train a regressor and log its metrics.
4. Register the model `airqmodelfcd91b`.

---

### Step 1: Read Features from Feature Group

```python
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import SparkSession

# Initialize the Feature Engineering Client
fe = FeatureEngineeringClient()

# Define the feature table name
schema = dbutils.widgets.get("schema")
feature_table_name = f"{schema}.airqfcd91b"

# Read features from the feature group
feature_df = fe.read_table(name=feature_table_name)
feature_df.display()
```

---

### Step 2: Assemble Training Dataset

```python
from pyspark.sql.functions import col

# Select features and target for training
training_df = feature_df.select(
    "date",
    "pm25_lag1",
    "pm25_lag7",
    "pm25_rolling_avg_7d",
    "temp_humidity_ratio",
    "temperature",
    "humidity",
    "wind_speed",
    "pressure",
    "precipitation",
    "pm25"
)

# Write the training dataset to a table
training_table_name = f"{schema}.airqtdfcd91b"
training_df.write.saveAsTable(training_table_name)

print(f"Training dataset {training_table_name} created successfully.")
```

---

### Step 3: Train a Regressor and Log Metrics

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import pandas as pd

# Convert Spark DataFrame to Pandas
pandas_df = training_df.toPandas()

# Split into features and target
X = pandas_df.drop(["date", "pm25"], axis=1)
y = pandas_df["pm25"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")

# Log the model and metrics with MLflow
with mlflow.start_run():
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")
    mlflow.set_tag("schema", schema)

print("Model trained and logged successfully.")
```

---

### Step 4: Register the Model

```python
# Register the model in MLflow
model_name = "airqmodelfcd91b"
model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
mlflow.register_model(model_uri, model_name)

print(f"Model {model_name} registered successfully.")
```