# Databricks notebook source
# MAGIC %md
# Train Fraud Detection Model

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql.functions import col
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import mlflow
import pandas as pd

# Initialize Feature Engineering Client
fe = FeatureEngineeringClient()

# COMMAND ----------

# Read feature table
feature_table_name = f"workspace.{dbutils.widgets.get('schema')}.cctxne0b071"
features_df = fe.read_table(name=feature_table_name)

# COMMAND ----------

# Split data into train and test
train_df = features_df.filter(col("datetime") < "2025-12-01")
pandas_train = train_df.toPandas()

X_train = pandas_train[["txn_velocity_1h", "avg_amount_1h", "std_amount_1h", "geo_distance"]]
y_train = pandas_train["is_fraud"]

# COMMAND ----------

# Train model
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log model and parameters
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    
    # Evaluate on training data (for demonstration)
    y_pred_proba = model.predict_proba(X_train)[:, 1]
    auc = roc_auc_score(y_train, y_pred_proba)
    mlflow.log_metric("train_roc_auc", auc)
    
    # Register model
    model_name = f"workspace.{dbutils.widgets.get('schema')}.ccmodele0b071"
    mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", model_name)

# COMMAND ----------

# Log feature table and model in MLflow
with mlflow.start_run():
    mlflow.log_param("feature_table_name", feature_table_name)
    mlflow.log_param("model_name", model_name)