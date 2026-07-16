from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark

print("Starting prediction pipeline...")

catalog, schema = 'workspace', 'mlpabfcf9c1'

# Read from existing feature table
spark_df = spark.table(f"{catalog}.{schema}.cctxn015310")
print(f"Read {spark_df.count()} rows from feature table")

# Select features for training
feature_cols = ['amount', 'amount_log', 'hour_of_day', 'day_of_week', 'lat', 'long', 'category_index']

for c in feature_cols:
    spark_df = spark_df.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(spark_df)

train_data, val_data = df_assembled.randomSplit([0.8, 0.2], seed=42)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="is_fraud",
    numTrees=100,
    maxDepth=10,
    seed=42
)

print("Training...")
rf_model = rf.fit(train_data)
print("Training complete")

val_predictions = rf_model.transform(val_data)

evaluator = BinaryClassificationEvaluator(
    labelCol="is_fraud",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

val_auc = evaluator.evaluate(val_predictions)
print(f"Validation ROC AUC: {val_auc:.4f}")

# Register model
mlflow.set_experiment(f"/Users/benedict@hopsworks.ai/mlpabfcf9c1/ccfraud_experiment")

with mlflow.start_run():
    mlflow.log_param("num_trees", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("val_roc_auc", val_auc)
    mlflow.spark.log_model(rf_model, "model")
    
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mv = mlflow.register_model(
        model_uri=model_uri,
        name=f"{catalog}.{schema}.ccmodel015310"
    )
    print(f"Model registered: {mv.name}, version: {mv.version}")

# Score transactions
volume_path = "/Volumes/workspace/mlpabfcf9c1/ccfraud_data/"
df_score = spark.read.csv(f"{volume_path}score_transactions.csv", header=True, inferSchema=True)
df_score = df_score.withColumn("datetime", F.to_timestamp("datetime", "yyyy-MM-dd'T'HH:mm:ss'Z'"))

df_score = df_score.withColumn("hour_of_day", F.hour("datetime"))
df_score = df_score.withColumn("day_of_week", F.dayofweek("datetime"))
df_score = df_score.withColumn("amount_log", F.log1p("amount"))

category_indexer = StringIndexer(inputCol="category", outputCol="category_index")
category_indexer_model = category_indexer.fit(df_score)
df_score = category_indexer_model.transform(df_score)

for c in feature_cols:
    df_score = df_score.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

df_score_assembled = assembler.transform(df_score)
score_predictions = rf_model.transform(df_score_assembled)

predictions_df = score_predictions.select(
    "transaction_id",
    F.col("probability")[1].alias("fraud_probability")
)

print(f"Predictions count: {predictions_df.count()}")

# Create predictions table
predictions_table_name = f"{catalog}.{schema}.ccpred015310"
predictions_df.write.mode("overwrite").saveAsTable(predictions_table_name)

# Publish for online lookup
try:
    from databricks.feature_store import FeatureStoreClient
    fs = FeatureStoreClient()
    fs.publish_table(name=f"{catalog}.{schema}.cctxn015310", online=True)
    fs.publish_table(name=predictions_table_name, online=True)
    print("Published tables for online lookup")
except Exception as e:
    print(f"Could not publish online: {e}")

print(f"PIPELINE COMPLETE - ROC AUC: {val_auc:.4f}")
