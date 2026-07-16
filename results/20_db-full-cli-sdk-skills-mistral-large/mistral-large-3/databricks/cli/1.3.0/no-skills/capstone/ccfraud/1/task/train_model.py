# Databricks notebook source
# MAGIC %md
# MAGIC ## Train Fraud Detection Model

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
import mlflow.spark

# Load data
df = spark.table("workspace.${MLPAB_DATABRICKS_SCHEMA}.transactions")

# Feature engineering
categorical_cols = ["merchant", "category"]
indexers = [StringIndexer(inputCol=col, outputCol=col+"_idx").fit(df) for col in categorical_cols]

assembler = VectorAssembler(
    inputCols=["amount", "lat", "long"] + [col+"_idx" for col in categorical_cols],
    outputCol="features"
)

# Train/test split
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Model
rf = RandomForestClassifier(labelCol="is_fraud", featuresCol="features", numTrees=100)

# Pipeline
pipeline = Pipeline(stages=indexers + [assembler, rf])

# Train
model = pipeline.fit(train)

# Evaluate
predictions = model.transform(test)
evaluator = BinaryClassificationEvaluator(labelCol="is_fraud", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)

# Log to MLflow
with mlflow.start_run():
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.spark.log_model(model, "model")
    mlflow.set_tag("model_name", "ccmodele0b071")

# Register model
model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
mlflow.register_model(model_uri, "ccmodele0b071")

print(f"Model trained and registered with ROC AUC: {roc_auc}")