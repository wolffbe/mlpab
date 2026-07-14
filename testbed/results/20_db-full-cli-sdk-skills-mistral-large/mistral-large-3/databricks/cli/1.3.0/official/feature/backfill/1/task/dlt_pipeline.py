# Databricks DLT pipeline to deduplicate and register the accounts feature table.
import dlt
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window

# Read the three batches from the volume.
batch_1 = spark.read.format("csv").option("header", "true").load("/Volumes/workspace/mlpab58df8e/task_volume/batch_1.csv")
batch_2 = spark.read.format("csv").option("header", "true").load("/Volumes/workspace/mlpab58df8e/task_volume/batch_2.csv")
batch_3 = spark.read.format("csv").option("header", "true").load("/Volumes/workspace/mlpab58df8e/task_volume/batch_3.csv")

# Union all batches.
all_batches = batch_1.union(batch_2).union(batch_3)

# Deduplicate: keep the latest revision per row_id (using updated_at).
window = Window.partitionBy("row_id").orderBy(col("updated_at").desc())
deduped = all_batches.withColumn("rank", row_number().over(window)).filter(col("rank") == 1).drop("rank")

# Register as a feature table in the target schema.
dlt.create_feature_table(
    name="workspace.mlpab58df8e.accounts7b3169",
    primary_keys=["row_id"],
    timestamp_keys=["updated_at"],
    schema=deduped.schema
)

# Write the deduplicated data to the feature table.
dlt.apply_changes(
    target="workspace.mlpab58df8e.accounts7b3169",
    source=deduped,
    keys=["row_id"],
    sequence_by="updated_at",
    apply_as_deletes=None,
    except_column_list=[],
    stored_as_scd_type=1
)