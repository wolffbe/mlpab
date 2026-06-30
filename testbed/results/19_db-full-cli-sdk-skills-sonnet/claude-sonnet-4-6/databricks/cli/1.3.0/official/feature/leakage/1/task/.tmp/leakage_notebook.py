# Databricks notebook source

import pandas as pd
import numpy as np
from scipy import stats

# COMMAND ----------
# Load data from volume
df = spark.read.csv(
    "/Volumes/workspace/mlpab6007c0/mlpab6007c0_data/training_data.csv",
    header=True, inferSchema=True
)
print(f"Row count: {df.count()}")
df.show(5)

# COMMAND ----------
# Convert to pandas for correlation analysis
pdf = df.toPandas()

features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']

print("=== Pearson Correlation with label (absolute) ===")
for col in features:
    corr = abs(pdf[col].corr(pdf['label']))
    print(f"  {col}: {corr:.4f}")

# COMMAND ----------
print("\n=== Point-biserial correlation ===")
results = {}
for col in features:
    corr, pval = stats.pointbiserialr(pdf['label'], pdf[col])
    results[col] = {'corr': abs(corr), 'pval': pval}
    print(f"  {col}: |corr|={abs(corr):.4f}, p-value={pval:.2e}")

print("\nSorted by absolute correlation:")
for col, vals in sorted(results.items(), key=lambda x: -x[1]['corr']):
    print(f"  {col}: |corr|={vals['corr']:.4f}")

# COMMAND ----------
# AUC for each feature alone using Decision Tree
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC')

print("\n=== AUC for each feature alone (Decision Tree depth=3) ===")
auc_results = {}
for col in features:
    assembler = VectorAssembler(inputCols=[col], outputCol='features')
    df_feat = assembler.transform(df)

    dt = DecisionTreeClassifier(featuresCol='features', labelCol='label', maxDepth=3)
    model = dt.fit(df_feat)
    preds = model.transform(df_feat)
    auc = evaluator.evaluate(preds)
    auc_results[col] = auc
    print(f"  {col}: AUC={auc:.4f}")

print("\nSorted by AUC:")
for col, auc in sorted(auc_results.items(), key=lambda x: -x[1]):
    print(f"  {col}: AUC={auc:.4f}")

best_feature = max(auc_results, key=lambda x: auc_results[x])
print(f"\nLeaking feature: {best_feature} (AUC={auc_results[best_feature]:.4f})")

# COMMAND ----------
# Save result to volume
import json
result = {
    "feature": best_feature,
    "evidence": f"AUC={auc_results[best_feature]:.4f} for {best_feature} alone (next best: {sorted(auc_results.values(), reverse=True)[1]:.4f})",
    "all_aucs": auc_results
}
print(json.dumps(result, indent=2))

# Write to volume for retrieval
with open('/tmp/leakage_result.json', 'w') as f:
    json.dump(result, f)

dbutils.fs.cp("file:///tmp/leakage_result.json", "/Volumes/workspace/mlpab6007c0/mlpab6007c0_data/leakage_result.json")
print("Result written to volume")
