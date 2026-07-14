# Databricks notebook to compute scores

# Step 1: Read the feature history
feature_history_df = spark.read.csv(
    "dbfs:/Volumes/workspace/mlpabbed188/scores4a1a3b_volume/feature_history.csv",
    header=True,
    inferSchema=True
)

# Step 2: Filter to the most recent revision at or before T
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

T = 1773234000000
window = Window.partitionBy("account_id").orderBy(feature_history_df["event_time"].desc())
latest_features_df = (
    feature_history_df
    .filter(feature_history_df["event_time"] <= T)
    .withColumn("rn", row_number().over(window))
    .filter("rn = 1")
    .drop("rn", "event_time")
)

# Step 3: Load the model
model_df = spark.read.json("dbfs:/Volumes/workspace/mlpabbed188/scores4a1a3b_volume/model.json")
weights = model_df.select("weights.*").first().asDict()
bias = model_df.select("bias").first()["bias"]

# Step 4: Compute the score
from pyspark.sql.functions import col, expr

scores_df = latest_features_df.withColumn(
    "score",
    expr(f"round(1.0 / (1.0 + exp(-({weights['f1']} * f1 + {weights['f2']} * f2 + {weights['f3']} * f3 + {bias}))), 6)")
)

# Step 5: Write the results to a Delta table
scores_df.select("account_id", "score").write.saveAsTable(
    "workspace.mlpabbed188.scores4a1a3b",
    mode="overwrite"
)

# Step 6: Enable online access for low-latency lookup
spark.sql("CREATE OR REPLACE ONLINE TABLE workspace.mlpabbed188.scores4a1a3b_online AS SELECT * FROM workspace.mlpabbed188.scores4a1a3b")