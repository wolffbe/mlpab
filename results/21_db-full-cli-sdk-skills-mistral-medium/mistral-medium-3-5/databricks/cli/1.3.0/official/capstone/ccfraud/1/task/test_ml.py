# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

print("Testing ML training...")

volume_path = "/Volumes/workspace/mlpabfcf9c1/ccfraud_data/"
transactions_path = f"{volume_path}transactions.csv"

df = spark.read.csv(transactions_path, header=True, inferSchema=True)
df = df.withColumn("datetime", F.to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ss'Z'"))

print(f"Initial count: {df.count()}")

# Add simple features
window_spec = Window.partitionBy("cc_num")
df = df.withColumn("hour_of_day", F.hour("datetime"))
df = df.withColumn("amount_log", F.log1p("amount"))
df = df.withColumn("amount_mean", F.mean("amount").over(window_spec))
df = df.withColumn("amount_std", F.stddev("amount").over(window_spec))
df = df.withColumn("amount_zscore", 
    (F.col("amount") - F.col("amount_mean")) / (F.col("amount_std") + 1e-6))

# Category encoding
category_indexer = StringIndexer(inputCol="category", outputCol="category_index")
df = category_indexer.fit(df).transform(df)

print(f"After features: {df.count()}")

# Replace nulls
feature_cols = ['amount', 'amount_log', 'amount_mean', 'amount_std', 'amount_zscore', 'hour_of_day', 'category_index']
for c in feature_cols:
    df = df.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df)

print(f"After assembly: {df_assembled.count()}")

# Split
train_data, val_data = df_assembled.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train_data.count()}, Val: {val_data.count()}")

# Train
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="is_fraud",
    numTrees=10,
    maxDepth=5,
    seed=42
)

print("Training...")
rf_model = rf.fit(train_data)
print("Training complete")

# Predict
val_predictions = rf_model.transform(val_data)

# Evaluate
evaluator = BinaryClassificationEvaluator(
    labelCol="is_fraud",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

val_auc = evaluator.evaluate(val_predictions)
print(f"Validation ROC AUC: {val_auc:.4f}")

print("ML test complete")
