# Databricks notebook source
# COMMAND ----------
df = spark.read.csv("/Volumes/workspace/mlpab97d2fb/leakage_vol/training_data.csv", header=True, inferSchema=True)
print(f"Row count: {df.count()}")
df.printSchema()

# COMMAND ----------
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, corr

features = ["f1", "f2", "f3", "f4", "f5", "f6"]
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

print("Per-feature AUC:")
results = {}
for feat in features:
    c = df.select(corr(col(feat), col("label").cast("double"))).collect()[0][0]
    assembler = VectorAssembler(inputCols=[feat], outputCol="features")
    assembled = assembler.transform(df)
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
    model = lr.fit(assembled)
    preds = model.transform(assembled)
    auc = evaluator.evaluate(preds)
    results[feat] = auc
    print(f"  {feat}: corr={c:.4f}, AUC={auc:.6f}")

leaking = max(results, key=results.get)
print(f"\nLeaking feature: {leaking} with AUC={results[leaking]:.6f}")

import json
result = {"feature": leaking, "evidence": {f: results[f] for f in features}}
result_str = json.dumps(result)
spark.createDataFrame([{"result": result_str}]).coalesce(1).write.mode("overwrite").text("/Volumes/workspace/mlpab97d2fb/leakage_vol/result_out")
print("Saved:", result_str)
