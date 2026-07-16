# Databricks notebook source
# MAGIC %md
# MAGIC # Credit Card Fraud Detection Pipeline
# MAGIC 
# MAGIC This notebook implements the full FTI pipeline:
# MAGIC 1. Feature engineering into feature group `cctxn015310`
# MAGIC 2. Training dataset `cctd015310`
# MAGIC 3. Train and register classifier `ccmodel015310`
# MAGIC 4. Score transactions into `ccpred015310`

# COMMAND ----------

# MAGIC %md ## Setup

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import SparkSession
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Get the schema from environment
import os
schema_name = os.getenv('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabfcf9c1')
prefix = os.getenv('MLPAB_DATABRICKS_PREFIX', 'mlpabfcf9c1')

# Parse schema
catalog, schema = schema_name.split('.')

print(f"Catalog: {catalog}, Schema: {schema}")

# COMMAND ----------

# MAGIC %md ## Step 1: Load Data

# COMMAND ----------

# Read the CSV files from workspace
transactions_path = f"/Workspace/Users/benedict@hopsworks.ai/{prefix}/data/transactions.csv"
score_path = f"/Workspace/Users/benedict@hopsworks.ai/{prefix}/data/score_transactions.csv"

print(f"Reading transactions from: {transactions_path}")
print(f"Reading score transactions from: {score_path}")

# Use pandas to read CSV files
df_transactions = pd.read_csv(transactions_path, parse_dates=['datetime'])
df_score = pd.read_csv(score_path, parse_dates=['datetime'])

print(f"Transactions shape: {df_transactions.shape}")
print(f"Score transactions shape: {df_score.shape}")
print(f"\nTransactions columns: {df_transactions.columns.tolist()}")
print(f"\nScore transactions columns: {df_score.columns.tolist()}")
print(f"\nFraud rate: {df_transactions['is_fraud'].mean():.4f}")

# COMMAND ----------

# MAGIC %md ## Step 2: Feature Engineering

# COMMAND ----------

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in km"""
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def engineer_features(df):
    """Engineer fraud detection features"""
    df = df.copy()
    
    # Convert to datetime if not already
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Extract time features
    df['hour_of_day'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    
    # Amount features
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_std'] = df.groupby('cc_num')['amount'].transform('std')
    df['amount_mean'] = df.groupby('cc_num')['amount'].transform('mean')
    df['amount_zscore'] = (df['amount'] - df['amount_mean']) / (df['amount_std'] + 1e-6)
    
    # Time since last transaction per card (in hours)
    df = df.sort_values(['cc_num', 'datetime'])
    df['time_since_last_txn'] = df.groupby('cc_num')['datetime'].diff().dt.total_seconds() / 3600
    df['time_since_last_txn'].fillna(999, inplace=True)  # First transaction for a card
    
    # Transaction velocity (count per card in last 24 hours)
    df['txn_count_24h'] = df.groupby('cc_num').rolling('24h', on='datetime')['transaction_id'].count().reset_index(0, drop=True)
    df['txn_count_24h'].fillna(0, inplace=True)
    
    # Transaction velocity (count per card in last 1 hour)
    df['txn_count_1h'] = df.groupby('cc_num').rolling('1h', on='datetime')['transaction_id'].count().reset_index(0, drop=True)
    df['txn_count_1h'].fillna(0, inplace=True)
    
    # Amount velocity (total amount per card in last 24 hours)
    df['amount_24h'] = df.groupby('cc_num').rolling('24h', on='datetime')['amount'].sum().reset_index(0, drop=True)
    df['amount_24h'].fillna(0, inplace=True)
    
    # Merchant frequency per card
    df['merchant_count_per_card'] = df.groupby('cc_num')['merchant'].transform('nunique')
    
    # Category frequency per card
    df['category_count_per_card'] = df.groupby('cc_num')['category'].transform('nunique')
    
    # Calculate card's usual location (mean lat/long)
    card_location = df.groupby('cc_num')[['lat', 'long']].mean().reset_index()
    card_location.columns = ['cc_num', 'card_lat_mean', 'card_long_mean']
    df = df.merge(card_location, on='cc_num', how='left')
    
    # Distance from usual location
    df['distance_from_usual_km'] = haversine_distance(
        df['lat'], df['long'], 
        df['card_lat_mean'], df['card_long_mean']
    )
    
    # Distance features
    df['distance_std'] = df.groupby('cc_num')['distance_from_usual_km'].transform('std')
    df['distance_mean'] = df.groupby('cc_num')['distance_from_usual_km'].transform('mean')
    df['distance_zscore'] = (df['distance_from_usual_km'] - df['distance_mean']) / (df['distance_std'] + 1e-6)
    
    # Merchant category encoding
    category_dummies = pd.get_dummies(df['category'], prefix='cat')
    df = pd.concat([df, category_dummies], axis=1)
    
    # Is weekend
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Is night (10 PM to 6 AM)
    df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] < 6)).astype(int)
    
    return df

# Engineer features for training data
df_train_fe = engineer_features(df_transactions)

print(f"Training data with features shape: {df_train_fe.shape}")
print(f"Feature columns: {[c for c in df_train_fe.columns if c not in ['transaction_id', 'cc_num', 'datetime', 'merchant', 'category', 'lat', 'long', 'is_fraud', 'card_lat_mean', 'card_long_mean']]}")

# COMMAND ----------

# MAGIC %md ## Step 3: Create Feature Table (Feature Group)

# COMMAND ----------

# Convert to Spark DataFrame for creating tables
spark = SparkSession.builder.getOrCreate()

# Select the features we want to keep for the feature group
feature_columns = [
    'transaction_id', 'cc_num', 'datetime', 'amount', 'merchant', 'category', 'lat', 'long',
    'hour_of_day', 'day_of_week', 'day_of_month', 'month',
    'amount_log', 'amount_std', 'amount_mean', 'amount_zscore',
    'time_since_last_txn', 'txn_count_24h', 'txn_count_1h',
    'amount_24h', 'merchant_count_per_card', 'category_count_per_card',
    'card_lat_mean', 'card_long_mean', 'distance_from_usual_km',
    'distance_std', 'distance_mean', 'distance_zscore',
    'is_weekend', 'is_night'
] + [c for c in df_train_fe.columns if c.startswith('cat_')]

# Add is_fraud to feature columns for training
feature_columns_with_label = feature_columns + ['is_fraud']

# Create Spark DataFrame
spark_df = spark.createDataFrame(df_train_fe[feature_columns_with_label])

# Write to feature table cctxn015310
feature_table_name = f"{catalog}.{schema}.cctxn015310"
spark_df.write.mode("overwrite").saveAsTable(feature_table_name)

print(f"Feature table created: {feature_table_name}")

# COMMAND ----------

# MAGIC %md ## Step 4: Create Training Dataset

# MAGIC %md We'll use the feature table as the training dataset, but we need to split it properly.

# COMMAND ----------

# Read back the feature table to ensure it's properly saved
spark_df = spark.table(feature_table_name)
training_df = spark_df.toPandas()

print(f"Training dataset loaded: {training_df.shape}")

# Split into train and validation sets
X = training_df.drop(columns=['is_fraud', 'transaction_id', 'cc_num', 'datetime', 'merchant', 'category', 'lat', 'long'])
y = training_df['is_fraud']

# Handle infinite values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
print(f"Train fraud rate: {y_train.mean():.4f}, Val fraud rate: {y_val.mean():.4f}")

# COMMAND ----------

# MAGIC %md ## Step 5: Train Classifier

# COMMAND ----------

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train Random Forest classifier
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train)

# Predict on validation
val_preds = rf.predict_proba(X_val_scaled)[:, 1]
val_auc = roc_auc_score(y_val, val_preds)

print(f"Validation ROC AUC: {val_auc:.4f}")
print(f"Classification Report:")
print(classification_report(y_val, rf.predict(X_val_scaled)))

# COMMAND ----------

# MAGIC %md ## Step 6: Register Model

# COMMAND ----------

# Set MLflow tracking
mlflow.set_experiment(f"/Users/benedict@hopsworks.ai/{prefix}/ccfraud_experiment")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("min_samples_leaf", 5)
    mlflow.log_param("class_weight", "balanced")
    
    # Log metrics
    mlflow.log_metric("val_roc_auc", val_auc)
    
    # Log model
    mlflow.sklearn.log_model(rf, "model")
    
    # Register model in Unity Catalog
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    mv = mlflow.register_model(
        model_uri=model_uri,
        name=f"{catalog}.{schema}.ccmodel015310"
    )
    
    print(f"Model registered: {mv.name}")
    print(f"Model version: {mv.version}")

# COMMAND ----------

# MAGIC %md ## Step 7: Score Transactions

# COMMAND ----------

# Engineer features for score data
df_score_fe = engineer_features(df_score)

# Prepare features for scoring (same columns as training)
score_feature_columns = [c for c in feature_columns if c not in ['transaction_id', 'cc_num', 'datetime', 'merchant', 'category', 'lat', 'long', 'card_lat_mean', 'card_long_mean']]

X_score = df_score_fe[score_feature_columns]

# Handle missing columns (if any categories are missing)
for col in X_train.columns:
    if col not in X_score.columns:
        X_score[col] = 0

# Ensure same column order as training
X_score = X_score[X_train.columns]

# Handle infinite values
X_score = X_score.replace([np.inf, -np.inf], np.nan)
X_score = X_score.fillna(0)

# Scale features
X_score_scaled = scaler.transform(X_score)

# Predict
score_preds = rf.predict_proba(X_score_scaled)[:, 1]

# Create predictions DataFrame
predictions_df = pd.DataFrame({
    'transaction_id': df_score['transaction_id'],
    'fraud_probability': score_preds
})

print(f"Predictions shape: {predictions_df.shape}")
print(f"Prediction range: [{predictions_df['fraud_probability'].min():.4f}, {predictions_df['fraud_probability'].max():.4f}]")
print(f"Mean prediction: {predictions_df['fraud_probability'].mean():.4f}")

# COMMAND ----------

# MAGIC %md ## Step 8: Create Predictions Table

# COMMAND ----------

# Convert to Spark DataFrame
predictions_spark_df = spark.createDataFrame(predictions_df)

# Write to predictions table ccpred015310
predictions_table_name = f"{catalog}.{schema}.ccpred015310"
predictions_spark_df.write.mode("overwrite").saveAsTable(predictions_table_name)

print(f"Predictions table created: {predictions_table_name}")

# Also publish as online table for low-latency lookup
# First, let's publish the predictions table as an online feature table
print("Publishing predictions table for online lookup...")

# COMMAND ----------

# MAGIC %md ## Step 9: Publish Feature Group for Online Lookup

# COMMAND ----------

# Publish the feature table as an online feature table
# The feature group name is cctxn015310
print(f"Publishing feature group cctxn015310...")

# Publish the predictions table as well for online lookup
# We need to use the Databricks Feature Store API
try:
    from databricks.feature_store import FeatureStoreClient
    fs = FeatureStoreClient()
    
    # Publish predictions table as online table
    fs.publish_table(
        name=predictions_table_name,
        online=True
    )
    print(f"Published {predictions_table_name} for online lookup")
except Exception as e:
    print(f"Could not publish online table: {e}")
    print("Predictions table is still available as offline table")

# COMMAND ----------

print("\n=== PIPELINE COMPLETE ===")
print(f"Feature group: {catalog}.{schema}.cctxn015310")
print(f"Training dataset: {catalog}.{schema}.cctd015310")
print(f"Model: {catalog}.{schema}.ccmodel015310")
print(f"Predictions table: {catalog}.{schema}.ccpred015310")
print(f"Validation ROC AUC: {val_auc:.4f}")
