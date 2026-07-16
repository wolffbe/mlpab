from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

print("Starting conservative full pipeline...")

catalog, schema = 'workspace', 'mlpabfcf9c1'

# Step 1: Read data and create feature table (already exists, but let's verify)
volume_path = "/Volumes/workspace/mlpabfcf9c1/ccfraud_data/"
df_transactions = spark.read.csv(f"{volume_path}transactions.csv", header=True, inferSchema=True)
df_score = spark.read.csv(f"{volume_path}score_transactions.csv", header=True, inferSchema=True)

print(f"Transactions: {df_transactions.count()}, Score: {df_score.count()}")

# Step 2: Simple feature engineering
df_transactions = df_transactions.withColumn("datetime", F.to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ss'Z'"))
df_score = df_score.withColumn("datetime", F.to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ss'Z'"))

df_transactions = df_transactions.withColumn("hour_of_day", F.hour("datetime"))
df_transactions = df_transactions.withColumn("day_of_week", F.dayofweek("datetime"))
df_transactions = df_transactions.withColumn("amount_log", F.log1p("amount"))

category_indexer = StringIndexer(inputCol="category", outputCol="category_index")
df_transactions = category_indexer.fit(df_transactions).transform(df_transactions)

# Step 3: Save feature table
feature_table_name = f"{catalog}.{schema}.cctxn015310"
df_transactions.write.mode("overwrite").saveAsTable(feature_table_name)

# Step 4: Save training dataset
training_table_name = f"{catalog}.{schema}.cctd015310"
df_transactions.write.mode("overwrite").saveAsTable(training_table_name)

print("Feature table and training dataset saved")

# Step 5: Train model
spark_df = spark.table(feature_table_name)

# Sample for faster training
spark_df = spark_df.sample(withReplacement=False, fraction=0.5, seed=42)

feature_cols = ['amount', 'amount_log', 'hour_of_day', 'day_of_week', 'lat', 'long', 'category_index']

for c in feature_cols:
    spark_df = spark_df.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(spark_df)

train_data, val_data = df_assembled.randomSplit([0.8, 0.2], seed=42)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="is_fraud",
    maxIter=50,
    regParam=0.01
)

print("Training...")
lr_model = lr.fit(train_data)
print("Training complete")

val_predictions = lr_model.transform(val_data)

evaluator = BinaryClassificationEvaluator(
    labelCol="is_fraud",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

val_auc = evaluator.evaluate(val_predictions)
print(f"Validation ROC AUC: {val_auc:.4f}")

# Step 6: Score transactions
df_score = df_score.withColumn("hour_of_day", F.hour("datetime"))
df_score = df_score.withColumn("day_of_week", F.dayofweek("datetime"))
df_score = df_score.withColumn("amount_log", F.log1p("amount"))

category_indexer_model = category_indexer.fit(df_score)
df_score = category_indexer_model.transform(df_score)

for c in feature_cols:
    df_score = df_score.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

df_score_assembled = assembler.transform(df_score)
score_predictions = lr_model.transform(df_score_assembled)

predictions_df = score_predictions.select(
    "transaction_id",
    F.col("probability")[1].alias("fraud_probability")
)

print(f"Predictions count: {predictions_df.count()}")

# Step 7: Save predictions
predictions_table_name = f"{catalog}.{schema}.ccpred015310"
predictions_df.write.mode("overwrite").saveAsTable(predictions_table_name)

print(f"\n=== PIPELINE COMPLETE ===")
print(f"Feature group: {catalog}.{schema}.cctxn015310")
print(f"Training dataset: {catalog}.{schema}.cctd015310")
print(f"Predictions table: {catalog}.{schema}.ccpred015310")
print(f"Validation ROC AUC: {val_auc:.4f}")
