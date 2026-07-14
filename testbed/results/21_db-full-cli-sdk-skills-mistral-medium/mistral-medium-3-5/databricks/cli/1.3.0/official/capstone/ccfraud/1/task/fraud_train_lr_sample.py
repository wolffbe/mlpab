from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark

print("Starting train with Logistic Regression on sample...")

catalog, schema = 'workspace', 'mlpabfcf9c1'

# Read from existing feature table
spark_df = spark.table(f"{catalog}.{schema}.cctxn015310")
print(f"Read {spark_df.count()} rows")

# Sample to 10000 rows to avoid memory issues
spark_df = spark_df.sample(withReplacement=False, fraction=0.5, seed=42)
print(f"Using {spark_df.count()} rows after sampling")

# Select features
feature_cols = ['amount', 'amount_log', 'hour_of_day', 'day_of_week', 'lat', 'long', 'category_index']

for c in feature_cols:
    spark_df = spark_df.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(spark_df)

train_data, val_data = df_assembled.randomSplit([0.8, 0.2], seed=42)

# Use Logistic Regression
lr = LogisticRegression(
    featuresCol="features",
    labelCol="is_fraud",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.8
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

# Register model
mlflow.set_experiment(f"/Users/benedict@hopsworks.ai/mlpabfcf9c1/ccfraud_experiment")

with mlflow.start_run():
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 100)
    mlflow.log_metric("val_roc_auc", val_auc)
    mlflow.spark.log_model(lr_model, "model")
    
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mv = mlflow.register_model(
        model_uri=model_uri,
        name=f"{catalog}.{schema}.ccmodel015310"
    )
    print(f"Model registered: {mv.name}, version: {mv.version}")

print(f"DONE - ROC AUC: {val_auc:.4f}")
