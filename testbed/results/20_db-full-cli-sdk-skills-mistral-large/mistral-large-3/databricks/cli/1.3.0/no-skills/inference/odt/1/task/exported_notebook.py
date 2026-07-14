# Databricks notebook source
# MAGIC %md
# Compute Scored Features

This notebook computes the `distance_deg` and `score` features for scoring requests.

---

## Step 1: Load Data

Load the requests and profiles data from the local `data/` directory.

```python
requests_df = spark.read.csv("/data/requests.csv", header=True, inferSchema=True)
profiles_df = spark.read.csv("/data/profiles.csv", header=True, inferSchema=True)
```

---

## Step 2: Join Data

Join the requests and profiles data on `account_id`.

```python
joined_df = requests_df.join(profiles_df, "account_id", "inner")
```

---

## Step 3: Compute Features

Compute the `distance_deg` and `score` features as specified.

```python
from pyspark.sql.functions import col, sqrt, round

# Compute distance_deg
joined_df = joined_df.withColumn(
    "distance_deg",
    round(
        sqrt(
            (col("request_lat") - col("home_lat")) ** 2 + 
            (col("request_lon") - col("home_lon")) ** 2
        ),
        6
    )
)

# Compute score
joined_df = joined_df.withColumn(
    "score",
    round(col("base_score") - 0.1 * col("distance_deg"), 6)
)
```

---

## Step 4: Select and Write Output

Select the required columns and write the result to a Delta table in the specified schema.

```python
output_df = joined_df.select("request_id", "account_id", "distance_deg", "score")

# Write to Delta table
output_table_name = "scoreda4f6e2"
output_schema = "${MLPAB_DATABRICKS_SCHEMA}"

output_df.write.format("delta").mode("overwrite").saveAsTable(f"{output_schema}.{output_table_name}")

# Enable online/real-time access
spark.sql(f"CREATE TABLE IF NOT EXISTS {output_schema}.{output_table_name}_online AS SELECT * FROM {output_schema}.{output_table_name} WITH (ENABLE ONLINE = TRUE);")
```