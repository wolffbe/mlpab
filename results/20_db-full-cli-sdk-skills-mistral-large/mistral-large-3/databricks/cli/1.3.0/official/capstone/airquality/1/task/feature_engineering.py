# Databricks notebook source
# MAGIC %md
# MAGIC ## Feature Engineering for Air Quality PM2.5 Forecasting

This notebook engineers features from the air quality history data and writes them to the feature group `airqfcd91b`.

### Steps:
1. Read the air quality history data.
2. Engineer features (e.g., rolling averages, lag features).
3. Write features to the feature group `airqfcd91b`.

---

### Step 1: Read the Air Quality History Data

```python
import pandas as pd
from pyspark.sql import SparkSession
from databricks.feature_engineering import FeatureEngineeringClient

# Read the air quality history data
history_df = pd.read_csv("/dbfs/data/airquality_history.csv")
spark_df = SparkSession.getActiveSession().createDataFrame(history_df)

# Display the data
spark_df.display()
```

---

### Step 2: Engineer Features

```python
from pyspark.sql.functions import col, lag, avg
from pyspark.sql.window import Window

# Define a window for rolling calculations
window = Window.orderBy("date").rowsBetween(-7, 0)

# Engineer features
feature_df = spark_df.withColumn("pm25_lag7", lag("pm25", 7).over(Window.orderBy("date")))
feature_df = feature_df.withColumn("pm25_rolling_avg_7d", avg("pm25").over(window))
feature_df = feature_df.withColumn("temp_humidity_ratio", col("temperature") / col("humidity"))

# Drop rows with null values (due to lag features)
feature_df = feature_df.na.drop()

# Display the engineered features
feature_df.display()
```

---

### Step 3: Write Features to Feature Group

```python
# Initialize the Feature Engineering Client
fe = FeatureEngineeringClient()

# Define the feature table name
schema = dbutils.widgets.get("schema")
feature_table_name = f"{schema}.airqfcd91b"

# Write features to the feature group
fe.create_table(
    name=feature_table_name,
    primary_keys=["date"],
    df=feature_df,
    description="Air quality features for PM2.5 forecasting"
)

print(f"Feature group {feature_table_name} created successfully.")
```