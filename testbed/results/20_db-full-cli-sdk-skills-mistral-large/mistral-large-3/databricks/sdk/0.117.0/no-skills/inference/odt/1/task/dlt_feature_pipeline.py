# Databricks notebook source
# MAGIC %md
# MAGIC ## Feature Table: scoreda4f6e2
# MAGIC 
This pipeline creates a feature table named `scoreda4f6e2` with columns: `request_id`, `account_id`, `distance_deg`, `score`.
Online access is enabled for low-latency lookup.

**Schema:** `{os.getenv('MLPAB_DATABRICKS_SCHEMA')}`

**Source Data:**
- `data/requests.csv` (requests)
- `data/profiles.csv` (profiles)

**Transformations:**
- `distance_deg = sqrt((request_lat - home_lat)^2 + (request_lon - home_lon)^2)` (rounded to 6 decimal places)
- `score = base_score - 0.1 * distance_deg` (rounded to 6 decimal places)

**Output:**
- Feature table: `scoreda4f6e2`
- Online table: Enabled

**Pipeline Name:** `{os.getenv('MLPAB_DATABRICKS_PREFIX')}_feature_pipeline`

---

**Note:** This notebook is intended to be run as a Delta Live Tables (DLT) pipeline.

COMMAND ----------

import dlt
from pyspark.sql.functions import col, sqrt, round
from pyspark.sql.types import DoubleType

# Read source data
requests_df = spark.read.csv("file:/Workspace/Users/{os.getenv('USER')}/{os.getenv('MLPAB_DATABRICKS_PREFIX')}/data/requests.csv", header=True, inferSchema=True)
profiles_df = spark.read.csv("file:/Workspace/Users/{os.getenv('USER')}/{os.getenv('MLPAB_DATABRICKS_PREFIX')}/data/profiles.csv", header=True, inferSchema=True)

# Join requests with profiles
merged_df = requests_df.join(profiles_df, "account_id", "left")

# Compute on-demand features
merged_df = merged_df.withColumn(
    "distance_deg", 
    round(
        sqrt((col("request_lat") - col("home_lat"))**2 + (col("request_lon") - col("home_lon"))**2),
        6
    )
)

merged_df = merged_df.withColumn(
    "score", 
    round(col("base_score") - 0.1 * col("distance_deg"), 6)
)

# Select required columns
result_df = merged_df.select("request_id", "account_id", "distance_deg", "score")

# Create the feature table
dlt.create_target_table(
    name="scoreda4f6e2",
    comment="Feature table for request scoring with distance-based adjustment.",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.managed": "true"
    }
)

# Write to the feature table
dlt.apply_changes(
    target="scoreda4f6e2",
    source="result_df",
    keys=["request_id"],
    sequence_by="request_id",
    apply_as_deletes=None,
    except_column_list=None,
    stored_as_scd_type=1
)

# Enable online access for low-latency lookup
spark.sql(f"""
CREATE OR REFRESH LIVE TABLE scoreda4f6e2 ONLINE TABLE
OPTIONS (
  primary_key = 'request_id'
)
AS SELECT * FROM {{{{ ref('scoreda4f6e2') }}}}
""")