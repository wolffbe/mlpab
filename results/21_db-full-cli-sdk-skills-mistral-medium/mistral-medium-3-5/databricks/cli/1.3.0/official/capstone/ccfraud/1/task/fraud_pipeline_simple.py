# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

print("Starting simple fraud pipeline...")

# Get the schema from environment
schema_name = 'workspace.mlpabfcf9c1'
catalog, schema = schema_name.split('.')

# Read the CSV files from the volume
volume_path = "/Volumes/workspace/mlpabfcf9c1/ccfraud_data/"
transactions_path = f"{volume_path}transactions.csv"
score_path = f"{volume_path}score_transactions.csv"

print(f"Reading transactions from: {transactions_path}")

df_transactions = spark.read.csv(transactions_path, header=True, inferSchema=True)
df_score = spark.read.csv(score_path, header=True, inferSchema=True)

# Convert datetime string to timestamp
df_transactions = df_transactions.withColumn("datetime", F.to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ss'Z'"))
df_score = df_score.withColumn("datetime", F.to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ss'Z'"))

print(f"Transactions count: {df_transactions.count()}")
print(f"Score transactions count: {df_score.count()}")

# Simple feature engineering (no window functions)
df_transactions = df_transactions.withColumn("hour_of_day", F.hour("datetime"))
df_transactions = df_transactions.withColumn("day_of_week", F.dayofweek("datetime"))
df_transactions = df_transactions.withColumn("amount_log", F.log1p("amount"))

# Category encoding
category_indexer = StringIndexer(inputCol="category", outputCol="category_index")
df_transactions = category_indexer.fit(df_transactions).transform(df_transactions)

print(f"Feature engineering complete")

# Create feature table
feature_table_name = f"{catalog}.{schema}.cctxn015310"
df_transactions.write.mode("overwrite").saveAsTable(feature_table_name)

# Create training dataset
training_table_name = f"{catalog}.{schema}.cctd015310"
df_transactions.write.mode("overwrite").saveAsTable(training_table_name)

print(f"Tables created")

# Train classifier
spark_df = spark.table(feature_table_name)

feature_cols = ['amount', 'amount_log', 'hour_of_day', 'day_of_week', 'lat', 'long', 'category_index']

for c in feature_cols:
    spark_df = spark_df.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
assembler_model = assembler.fit(spark_df)
df_assembled = assembler_model.transform(spark_df)

train_data, val_data = df_assembled.randomSplit([0.8, 0.2], seed=42)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="is_fraud",
    numTrees=100,
    maxDepth=10,
    minInstancesPerNode=5,
    seed=42,
    subsamplingRate=0.8
)

rf_model = rf.fit(train_data)
val_predictions = rf_model.transform(val_data)

evaluator = BinaryClassificationEvaluator(
    labelCol="is_fraud",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

val_auc = evaluator.evaluate(val_predictions)
print(f"Validation ROC AUC: {val_auc:.4f}")

# Score transactions
df_score = df_score.withColumn("hour_of_day", F.hour("datetime"))
df_score = df_score.withColumn("day_of_week", F.dayofweek("datetime"))
df_score = df_score.withColumn("amount_log", F.log1p("amount"))

category_indexer_model = category_indexer.fit(df_score)
df_score = category_indexer_model.transform(df_score)

for c in feature_cols:
    df_score = df_score.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

df_score_assembled = assembler_model.transform(df_score)
score_predictions = rf_model.transform(df_score_assembled)

predictions_df = score_predictions.select(
    "transaction_id",
    F.col("probability")[1].alias("fraud_probability")
)

print(f"Predictions count: {predictions_df.count()}")

# Create predictions table
predictions_table_name = f"{catalog}.{schema}.ccpred015310"
predictions_df.write.mode("overwrite").saveAsTable(predictions_table_name)

print(f"\n=== PIPELINE COMPLETE ===")
print(f"Feature group: {catalog}.{schema}.cctxn015310")
print(f"Training dataset: {catalog}.{schema}.cctd015310")
print(f"Predictions table: {catalog}.{schema}.ccpred015310")
print(f"Validation ROC AUC: {val_auc:.4f}")
