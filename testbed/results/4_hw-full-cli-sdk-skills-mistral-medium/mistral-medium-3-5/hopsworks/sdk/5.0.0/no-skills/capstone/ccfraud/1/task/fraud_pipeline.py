#!/usr/bin/env python3
"""
Credit card fraud detection pipeline.
This script runs ON THE PLATFORM and does all the work there.
"""

import hopsworks
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os

# Connect to Hopsworks (this will work on the platform)
proj = hopsworks.login()
fs = proj.get_feature_store()

# Read the data from Resources
print("Reading data from Resources...")
ds_api = proj.get_dataset_api()

# Download transactions data
local_transactions_path = ds_api.download("Resources/transactions.csv/transactions.csv", overwrite=True)
transactions_df = pd.read_csv(local_transactions_path)

# Download score data
local_score_path = ds_api.download("Resources/score_transactions.csv/score_transactions.csv", overwrite=True)
score_df = pd.read_csv(local_score_path)

print(f"Transactions: {len(transactions_df)} rows")
print(f"Score transactions: {len(score_df)} rows")

# Feature Engineering
print("\nEngineering features...")

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
    
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    
    # Time since first transaction per card
    df['txn_order'] = df.groupby('cc_num').cumcount()
    df['time_since_first_txn'] = df.groupby('cc_num')['datetime'].transform(
        lambda x: (x - x.min()).dt.total_seconds() / 3600  # hours
    )
    
    # Transaction velocity features
    df['amount_per_hour'] = df['amount'] / (df['time_since_first_txn'] + 1)
    
    # Rolling window features (last 5 transactions per card)
    df['rolling_avg_amount_5'] = df.groupby('cc_num')['amount'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df['rolling_std_amount_5'] = df.groupby('cc_num')['amount'].transform(
        lambda x: x.rolling(5, min_periods=1).std()
    )
    df['rolling_count_5'] = df.groupby('cc_num')['amount'].transform(
        lambda x: x.rolling(5, min_periods=1).count()
    )
    
    # Time since last transaction per card
    df['time_since_last_txn'] = df.groupby('cc_num')['datetime'].diff().dt.total_seconds() / 3600
    df['time_since_last_txn'] = df['time_since_last_txn'].fillna(999)  # First txn
    
    # Card-level aggregates
    df['card_avg_amount'] = df.groupby('cc_num')['amount'].transform('mean')
    df['card_std_amount'] = df.groupby('cc_num')['amount'].transform('std')
    df['card_total_amount'] = df.groupby('cc_num')['amount'].transform('sum')
    df['card_txn_count'] = df.groupby('cc_num')['amount'].transform('count')
    
    # Category encoding
    category_dummies = pd.get_dummies(df['category'], prefix='cat')
    for col in category_dummies.columns:
        df[col] = category_dummies[col].astype(int)
    
    # Merchant frequency
    df['merchant_freq'] = df.groupby('merchant')['merchant'].transform('count')
    
    # Geo features - calculate distance from card's mean location
    df['card_mean_lat'] = df.groupby('cc_num')['lat'].transform('mean')
    df['card_mean_long'] = df.groupby('cc_num')['long'].transform('mean')
    df['card_std_lat'] = df.groupby('cc_num')['lat'].transform('std')
    df['card_std_long'] = df.groupby('cc_num')['long'].transform('std')
    
    # Distance from card's mean location
    df['distance_from_mean_km'] = haversine_distance(
        df['lat'], df['long'], 
        df['card_mean_lat'], df['card_mean_long']
    )
    
    # Distance from previous transaction location (per card)
    df['prev_lat'] = df.groupby('cc_num')['lat'].shift(1)
    df['prev_long'] = df.groupby('cc_num')['long'].shift(1)
    df['distance_from_prev_km'] = haversine_distance(
        df['lat'], df['long'], 
        df['prev_lat'], df['prev_long']
    )
    df['distance_from_prev_km'] = df['distance_from_prev_km'].fillna(0)
    
    # Amount ratios
    df['amount_ratio_to_avg'] = df['amount'] / (df['card_avg_amount'] + 0.01)
    df['amount_ratio_to_std'] = df['amount'] / (df['card_std_amount'] + 0.01)
    
    # High-risk signals
    df['is_high_amount'] = (df['amount'] > df['card_avg_amount'] * 3).astype(int)
    df['is_far_from_mean'] = (df['distance_from_mean_km'] > 100).astype(int)
    df['is_rapid_txn'] = (df['time_since_last_txn'] < 1).astype(int)
    
    return df

# Engineer features for training data
print("Engineering training features...")
engineered_train = engineer_features(transactions_df)

# Select features to use
feature_columns = [
    'amount', 'hour', 'day_of_week', 'day_of_month', 'month',
    'txn_order', 'time_since_first_txn', 'amount_per_hour',
    'rolling_avg_amount_5', 'rolling_std_amount_5', 'rolling_count_5',
    'time_since_last_txn', 'card_avg_amount', 'card_std_amount',
    'card_total_amount', 'card_txn_count', 'merchant_freq',
    'card_mean_lat', 'card_mean_long', 'card_std_lat', 'card_std_long',
    'distance_from_mean_km', 'distance_from_prev_km',
    'amount_ratio_to_avg', 'amount_ratio_to_std',
    'is_high_amount', 'is_far_from_mean', 'is_rapid_txn'
]

# Add category one-hot columns
cat_cols = [c for c in engineered_train.columns if c.startswith('cat_')]
feature_columns.extend(cat_cols)

print(f"Total features: {len(feature_columns)}")

# Create feature group cctxnee3558
print("\nCreating feature group cctxnee3558...")
fg_name = "cctxnee3558"
fg_version = 1

# Delete existing feature group if it exists
try:
    existing_fg = fs.get_feature_group(fg_name, version=fg_version)
    if existing_fg is not None:
        existing_fg.delete()
        print(f"Deleted existing feature group {fg_name} v{fg_version}")
except:
    pass

fg = fs.get_or_create_feature_group(
    name=fg_name,
    version=fg_version,
    description="Credit card fraud detection features",
    primary_key=["transaction_id"],
    event_time="datetime"
)
print(f"Feature group {fg_name} v{fg_version} ready")

# Prepare data for feature group insertion
train_fg_data = engineered_train[['transaction_id', 'datetime'] + feature_columns + ['is_fraud']].copy()

# Convert int columns to int64 to match bigint
int_cols = [c for c in train_fg_data.columns if train_fg_data[c].dtype in ['int32', 'int64']]
for col in int_cols:
    train_fg_data[col] = train_fg_data[col].astype('int64')

# Insert into feature group
print("Inserting training data into feature group...")
fg.insert(train_fg_data, write_options={"wait_for_job": True})
print("Training data inserted into feature group")

# Create training dataset cctdee3558
print("\nCreating training dataset cctdee3558...")
dataset_name = "cctdee3558"

# Delete existing feature view if it exists
try:
    existing_fv = fs.get_feature_view(dataset_name, version=1)
    if existing_fv is not None:
        existing_fv.delete()
        print(f"Deleted existing feature view {dataset_name} v1")
except:
    pass

# Create a feature view for training
query = fg.select_all()
fv = fs.get_or_create_feature_view(
    name=dataset_name,
    version=1,
    query=query,
    labels=["is_fraud"]
)
print(f"Feature view {dataset_name} ready")

# Create training data
print("\nCreating training data...")
fv.training_data(
    description="Credit card fraud training data",
    data_format="csv",
    coalesce=False
)

# Get the training data for model training
print("\nFetching training data...")
features_df, labels_df = fv.get_training_data(training_dataset_version=1, dataframe_type="pandas")
train_df = features_df.copy()
train_df['is_fraud'] = labels_df['is_fraud']

print(f"Training data shape: {train_df.shape}")
print(f"Label distribution:\n{train_df['is_fraud'].value_counts()}")

# Prepare features and labels
X_cols = [c for c in train_df.columns if c not in ['transaction_id', 'datetime', 'is_fraud']]
X = train_df[X_cols]
y = train_df['is_fraud']

print(f"Features: {len(X_cols)}")
print(f"Samples: {len(X)}")

# Handle missing values
X = X.fillna(X.mean(numeric_only=True))

# Train a model
print("\nTraining model...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model_rf.fit(X_train, y_train)

# Predict on validation
val_preds = model_rf.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_preds)

print(f"Validation ROC AUC: {val_auc:.4f}")

# Register model ccmodelee3558
print("\nRegistering model ccmodelee3558...")
model_name = "ccmodelee3558"

mr = proj.get_model_registry()

# Save model locally first
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "model.pkl")
with open(model_path, 'wb') as f:
    pickle.dump(model_rf, f)

# Register model
metrics = {
    "roc_auc": val_auc,
    "accuracy": accuracy_score(y_val, model_rf.predict(X_val)),
    "precision": precision_score(y_val, model_rf.predict(X_val)),
    "recall": recall_score(y_val, model_rf.predict(X_val)),
    "f1": f1_score(y_val, model_rf.predict(X_val))
}

try:
    model_in_registry = mr.get_model(model_name)
    print(f"Model {model_name} already exists, updating...")
    model_in_registry.update(
        model_path=model_dir,
        description="Credit card fraud detection Random Forest classifier",
        metrics=metrics
    )
except Exception as e:
    print(f"Creating new model: {e}")
    # Create new model using sklearn API
    model_in_registry = mr.sklearn.create_model(
        name=model_name,
        metrics=metrics,
        description="Credit card fraud detection Random Forest classifier",
        model_data=model_dir
    )

print(f"Model {model_name} registered with ROC AUC: {val_auc:.4f}")

# Now score the score_transactions.csv
print("\nScoring test transactions...")

# Engineer features for score data
engineered_score = engineer_features(score_df)

# Select same features
score_features = engineered_score[['transaction_id'] + feature_columns].copy()

# Align columns with training
missing_cols = [c for c in X_cols if c not in score_features.columns]
for col in missing_cols:
    if col.startswith('cat_'):
        score_features[col] = 0
    else:
        score_features[col] = 0

# Reorder columns to match training
expected_cols = X_cols
score_features = score_features[['transaction_id'] + expected_cols]

# Fill missing values
X_score = score_features[expected_cols].fillna(X.mean(numeric_only=True))

# Predict
score_preds = model_rf.predict_proba(X_score)[:, 1]

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'transaction_id': score_features['transaction_id'],
    'fraud_probability': score_preds
})

print(f"Predictions shape: {predictions_df.shape}")
print(f"Prediction range: [{predictions_df['fraud_probability'].min():.4f}, {predictions_df['fraud_probability'].max():.4f}]")

# Create predictions feature table ccpredee3558
print("\nCreating predictions feature table ccpredee3558...")
pred_fg_name = "ccpredee3558"

# Delete existing predictions feature group if it exists
try:
    existing_pred_fg = fs.get_feature_group(pred_fg_name, version=1)
    if existing_pred_fg is not None:
        existing_pred_fg.delete()
        print(f"Deleted existing predictions feature group {pred_fg_name} v1")
except:
    pass

pred_fg = fs.get_or_create_feature_group(
    name=pred_fg_name,
    version=1,
    description="Credit card fraud predictions",
    primary_key=["transaction_id"],
    online_enabled=True  # Enable for low-latency lookup
)
print(f"Predictions feature group {pred_fg_name} ready")

# Insert predictions
print("Inserting predictions into feature table...")
pred_fg.insert(predictions_df, write_options={"wait_for_job": True})
print("Predictions inserted")

# Also create a feature view for online serving
print("\nCreating feature view for online serving...")
try:
    fv = fs.get_or_create_feature_view(
        name=f"{pred_fg_name}_view",
        version=1,
        query=pred_fg.select_all(),
        labels=["fraud_probability"]
    )
    print(f"Feature view created for online serving")
except Exception as e:
    print(f"Feature view creation: {e}")

print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
print(f"Feature group: {fg_name}")
print(f"Training dataset: {dataset_name}")
print(f"Model: {model_name} (ROC AUC: {val_auc:.4f})")
print(f"Predictions table: {pred_fg_name}")
print(f"Predictions range: [{predictions_df['fraud_probability'].min():.4f}, {predictions_df['fraud_probability'].max():.4f}]")
