#!/usr/bin/env python3
"""
Training and scoring script to be run as a Hopsworks job.
This script does feature engineering, model training, and scoring.
"""

import hopsworks
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import sys

# Connect to Hopsworks
hopsworks.login()
project = hopsworks.get_current_project()
fs = project.get_feature_store()
mr = project.get_model_registry()

print("=== Starting Training and Scoring Job ===")

# ============================================================================
# Step 1: Read raw data from files
# ============================================================================
print("\n=== Step 1: Reading raw data ===")

# Read training data - use the path where files were uploaded
# The files are in /Resources/apps/ (project's Resources/apps directory)
# Try multiple possible paths
possible_dirs = [
    '/Resources/apps/',
    '/Resources/',
    '/hopsfs/Resources/apps/',
    '/hopsfs/Resources/',
    f'/Projects/{project.name}/Resources/apps/',
    f'/Projects/{project.name}/Resources/',
]

transactions_path = None
for dir_path in possible_dirs:
    test_path = os.path.join(dir_path, 'transactions.csv')
    if os.path.exists(test_path):
        transactions_path = test_path
        break

if transactions_path is None:
    # Try to find it by walking the filesystem
    for root, dirs, files in os.walk('/'):
        if 'transactions.csv' in files:
            transactions_path = os.path.join(root, 'transactions.csv')
            break

if transactions_path is None:
    raise FileNotFoundError("Could not find transactions.csv")

# Find score_transactions.csv in the same directory
score_dir = os.path.dirname(transactions_path)
score_path = os.path.join(score_dir, 'score_transactions.csv')

print(f"Reading from: {transactions_path}")
print(f"Reading from: {score_path}")

# Read training data
transactions_df = pd.read_csv(transactions_path)
score_df = pd.read_csv(score_path)

print(f"Training transactions: {len(transactions_df)}")
print(f"Score transactions: {len(score_df)}")

# Parse datetime
train_df = transactions_df.copy()
train_df['datetime'] = pd.to_datetime(train_df['datetime'])
score_df['datetime'] = pd.to_datetime(score_df['datetime'])

# ============================================================================
# Step 2: Feature Engineering
# ============================================================================
print("\n=== Step 2: Feature Engineering ===")

def engineer_features(df, card_stats=None, is_training=True):
    """Engineer fraud detection features."""
    df = df.copy()
    
    # Basic features
    df['hour_of_day'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    
    # Amount features
    df['amount_log'] = np.log1p(df['amount'])
    
    if is_training:
        # Compute card statistics from training data
        df['amount_std'] = df.groupby('cc_num')['amount'].transform('std')
        df['amount_mean'] = df.groupby('cc_num')['amount'].transform('mean')
        df['amount_zscore'] = (df['amount'] - df['amount_mean']) / (df['amount_std'] + 1e-6)
        
        # Time-based features (velocity)
        df['time_since_last_tx'] = df.groupby('cc_num')['datetime'].diff().dt.total_seconds() / 3600  # hours
        # Use integer window sizes for compatibility
        df['tx_count_last_5'] = df.groupby('cc_num')['datetime'].transform(
            lambda x: x.rolling(5, min_periods=1).count()
        )
        df['tx_count_last_20'] = df.groupby('cc_num')['datetime'].transform(
            lambda x: x.rolling(20, min_periods=1).count()
        )
        df['amount_last_5'] = df.groupby('cc_num')['amount'].transform(
            lambda x: x.rolling(5, min_periods=1).sum()
        )
        df['amount_last_20'] = df.groupby('cc_num')['amount'].transform(
            lambda x: x.rolling(20, min_periods=1).sum()
        )
        
        # Merchant and category features
        merchant_counts = df.groupby('merchant').size()
        df['merchant_freq'] = df['merchant'].map(lambda x: merchant_counts.get(x, 0) / len(df))
        
        category_counts = df.groupby('category').size()
        df['category_freq'] = df['category'].map(lambda x: category_counts.get(x, 0) / len(df))
        
        # Card-merchant frequency
        cc_merchant_counts = df.groupby(['cc_num', 'merchant']).size()
        cc_counts = df.groupby('cc_num').size()
        df['cc_merchant_freq'] = df.apply(lambda row: cc_merchant_counts.get((row['cc_num'], row['merchant']), 0) / max(cc_counts.get(row['cc_num'], 1), 1), axis=1)
        
        # Geo features
        df['lat_long_ratio'] = df['lat'] / (df['long'].abs() + 1e-6)
        
        # Card-level statistics
        df['cc_tx_count'] = df.groupby('cc_num').cumcount()
        df['cc_avg_amount'] = df.groupby('cc_num')['amount'].transform('mean')
        df['cc_std_amount'] = df.groupby('cc_num')['amount'].transform('std')
        
        # Category amount stats per card
        df['cc_cat_avg'] = df.groupby(['cc_num', 'category'])['amount'].transform('mean')
        df['cc_cat_std'] = df.groupby(['cc_num', 'category'])['amount'].transform('std')
        
        # Time since first transaction for this card
        df['time_since_first_tx'] = (df['datetime'] - df.groupby('cc_num')['datetime'].transform('min')).dt.total_seconds() / 3600
        
        # Rolling average amount per card
        df['rolling_avg_amount'] = df.groupby('cc_num')['amount'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        
        # Rolling std amount per card
        df['rolling_std_amount'] = df.groupby('cc_num')['amount'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        )
        
        # Geo distance from card's mean location
        card_mean_loc = df.groupby('cc_num')[['lat', 'long']].transform('mean')
        df['lat_diff'] = df['lat'] - card_mean_loc['lat']
        df['long_diff'] = df['long'] - card_mean_loc['long']
        df['geo_distance'] = np.sqrt(df['lat_diff']**2 + df['long_diff']**2)
        
        # Interaction features
        df['amount_per_hour'] = df['amount'] / (df['time_since_last_tx'] + 1e-6)
        df['amount_to_avg_ratio'] = df['amount'] / (df['cc_avg_amount'] + 1e-6)
        
        # Fill NaN values
        df = df.fillna({
            'time_since_last_tx': 999,
            'tx_count_last_5': 0,
            'tx_count_last_20': 0,
            'amount_last_5': 0,
            'amount_last_20': 0,
            'amount_std': 0,
            'amount_mean': 0,
            'amount_zscore': 0,
            'cc_merchant_freq': 0,
            'cc_cat_avg': 0,
            'cc_cat_std': 0,
            'time_since_first_tx': 0,
            'rolling_avg_amount': 0,
            'rolling_std_amount': 0,
            'amount_per_hour': 0,
            'amount_to_avg_ratio': 0,
        })
        
        # Compute card stats for use with scoring data
        card_stats = df.groupby('cc_num').agg({
            'amount': ['mean', 'std'],
            'lat': 'mean',
            'long': 'mean',
        }).reset_index()
        card_stats.columns = ['cc_num', 'cc_avg_amount', 'cc_std_amount', 'cc_mean_lat', 'cc_mean_long']
        
        return df, card_stats
    else:
        # For scoring, use pre-computed card stats
        df = df.merge(card_stats, on='cc_num', how='left')
        
        # Compute features using card stats
        df['amount_std'] = 0  # Not available for scoring
        df['amount_mean'] = df['cc_avg_amount']
        df['amount_zscore'] = (df['amount'] - df['cc_avg_amount']) / (df['cc_std_amount'] + 1e-6)
        
        # Time-based features (will be 0 or default for scoring)
        df['time_since_last_tx'] = 999
        df['tx_count_last_5'] = 0
        df['tx_count_last_20'] = 0
        df['amount_last_5'] = 0
        df['amount_last_20'] = 0
        
        # Merchant and category features (use training stats)
        # These will be approximate for scoring
        df['merchant_freq'] = 0
        df['category_freq'] = 0
        df['cc_merchant_freq'] = 0
        
        # Geo features
        df['lat_long_ratio'] = df['lat'] / (df['long'].abs() + 1e-6)
        
        # Card-level statistics
        df['cc_tx_count'] = 0
        df['cc_cat_avg'] = 0
        df['cc_cat_std'] = 0
        
        # Time since first transaction
        df['time_since_first_tx'] = 0
        
        # Rolling features
        df['rolling_avg_amount'] = df['cc_avg_amount']
        df['rolling_std_amount'] = df['cc_std_amount']
        
        # Geo distance from card's mean location
        df['lat_diff'] = df['lat'] - df['cc_mean_lat']
        df['long_diff'] = df['long'] - df['cc_mean_long']
        df['geo_distance'] = np.sqrt(df['lat_diff']**2 + df['long_diff']**2)
        
        # Interaction features
        df['amount_per_hour'] = df['amount'] / 999  # Default time
        df['amount_to_avg_ratio'] = df['amount'] / (df['cc_avg_amount'] + 1e-6)
        
        # Fill remaining NaN
        df = df.fillna({
            'cc_avg_amount': 0,
            'cc_std_amount': 0,
            'cc_mean_lat': 0,
            'cc_mean_long': 0,
        })
        
        return df, card_stats

# Engineer features for training data
train_fe, card_stats = engineer_features(train_df, is_training=True)

# Engineer features for scoring data using training card stats
score_fe, _ = engineer_features(score_df, card_stats=card_stats, is_training=False)

print(f"Training features shape: {train_fe.shape}")
print(f"Scoring features shape: {score_fe.shape}")

# ============================================================================
# Step 3: Create/Update Feature Group with Engineered Features
# ============================================================================
print("\n=== Step 3: Creating/Updating Feature Group ===")

fg_name = "cctxnee3558"

# Check if feature group already exists
try:
    fg = fs.get_feature_group(fg_name, version=1)
    print(f"Feature group {fg_name} already exists")
    if fg is None:
        raise Exception("Feature group is None")
except Exception as e:
    print(f"Feature group does not exist or error: {e}")
    from hsfs import feature
    
    # Define features
    features = [
        feature.Feature("transaction_id", "string", primary=True),
        feature.Feature("cc_num", "string"),
        feature.Feature("datetime", "timestamp"),
        feature.Feature("amount", "float"),
        feature.Feature("merchant", "string"),
        feature.Feature("category", "string"),
        feature.Feature("lat", "float"),
        feature.Feature("long", "float"),
        feature.Feature("hour_of_day", "int"),
        feature.Feature("day_of_week", "int"),
        feature.Feature("day_of_month", "int"),
        feature.Feature("month", "int"),
        feature.Feature("amount_log", "float"),
        feature.Feature("amount_std", "float"),
        feature.Feature("amount_mean", "float"),
        feature.Feature("amount_zscore", "float"),
        feature.Feature("time_since_last_tx", "float"),
        feature.Feature("tx_count_last_5", "int"),
        feature.Feature("tx_count_last_20", "int"),
        feature.Feature("amount_last_5", "float"),
        feature.Feature("amount_last_20", "float"),
        feature.Feature("merchant_freq", "float"),
        feature.Feature("category_freq", "float"),
        feature.Feature("cc_merchant_freq", "float"),
        feature.Feature("lat_long_ratio", "float"),
        feature.Feature("cc_tx_count", "int"),
        feature.Feature("cc_avg_amount", "float"),
        feature.Feature("cc_std_amount", "float"),
        feature.Feature("cc_cat_avg", "float"),
        feature.Feature("cc_cat_std", "float"),
        feature.Feature("time_since_first_tx", "float"),
        feature.Feature("rolling_avg_amount", "float"),
        feature.Feature("rolling_std_amount", "float"),
        feature.Feature("lat_diff", "float"),
        feature.Feature("long_diff", "float"),
        feature.Feature("geo_distance", "float"),
        feature.Feature("amount_per_hour", "float"),
        feature.Feature("amount_to_avg_ratio", "float"),
        feature.Feature("is_fraud", "int"),
    ]
    
    # Create feature group
    fg = fs.create_feature_group(
        name=fg_name,
        version=1,
        description="Credit card fraud detection features",
        primary_key=["transaction_id"],
        event_time="datetime",
        online_enabled=True,
        features=features,
    )
    print(f"Created feature group: {fg_name}")

# Insert training data
# Get feature columns from the feature group
if hasattr(fg, 'features') and fg.features is not None:
    feature_columns = [f.name for f in fg.features]
else:
    # If features are not available, use all columns from the dataframe
    # that match the expected feature names
    feature_columns = [
        "transaction_id", "cc_num", "datetime", "amount", "merchant", "category",
        "lat", "long", "hour_of_day", "day_of_week", "day_of_month", "month",
        "amount_log", "amount_std", "amount_mean", "amount_zscore",
        "time_since_last_tx", "tx_count_last_5", "tx_count_last_20",
        "amount_last_5", "amount_last_20", "merchant_freq", "category_freq",
        "cc_merchant_freq", "lat_long_ratio", "cc_tx_count", "cc_avg_amount",
        "cc_std_amount", "cc_cat_avg", "cc_cat_std", "time_since_first_tx",
        "rolling_avg_amount", "rolling_std_amount", "lat_diff", "long_diff",
        "geo_distance", "amount_per_hour", "amount_to_avg_ratio", "is_fraud"
    ]

train_insert_df = train_fe[feature_columns].copy()

print(f"Inserting {len(train_insert_df)} rows into feature group...")
fg.insert(train_insert_df, overwrite=True, wait=True)
print("Training data inserted successfully")

# ============================================================================
# Step 4: Create Training Dataset
# ============================================================================
print("\n=== Step 4: Creating Training Dataset ===")

td_name = "cctdee3558"

# Check if training dataset already exists
try:
    td = fs.get_training_dataset(td_name, version=1)
    print(f"Training dataset {td_name} already exists")
except:
    # Create training dataset from feature group
    query = fg.select_all()
    
    td = fs.create_training_dataset(
        name=td_name,
        version=1,
        description="Training dataset for credit card fraud detection",
        data_format="csv",
        query=query,
    )
    print(f"Created training dataset: {td_name}")

# ============================================================================
# Step 5: Train Model
# ============================================================================
print("\n=== Step 5: Training Model ===")

# Get the training dataset as a DataFrame
td_df = td.to_pandas()
print(f"Training dataset shape: {td_df.shape}")

# Prepare features and target
feature_cols = [col for col in td_df.columns if col not in ['transaction_id', 'is_fraud', 'datetime']]
X = td_df[feature_cols]
y = td_df['is_fraud']

print(f"Features: {len(feature_cols)}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
print("Training RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
print("Evaluating model...")
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]

roc_auc = roc_auc_score(y_val, y_pred_proba)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"Validation ROC AUC: {roc_auc:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1: {f1:.4f}")

metrics = {
    'roc_auc': roc_auc,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
}

# ============================================================================
# Step 6: Register Model
# ============================================================================
print("\n=== Step 6: Registering Model ===")

model_name = "ccmodelee3558"

# Save model
model_path = "/Resources/model.pkl"
joblib.dump(model, model_path)

# Register model using the Python API
python_model = mr.python.create_model(
    name=model_name,
    version=1,
    description="Credit card fraud detection RandomForest model",
    metrics=metrics,
    model=model,
    input_example=X_train.iloc[0:1].to_dict('records')[0],
    output_example={'prediction': 0, 'probability': [0.9, 0.1]},
)

# Save the model artifact
python_model.save(model_path)
print(f"Model {model_name} registered successfully with ROC AUC: {roc_auc:.4f}")

# ============================================================================
# Step 7: Score Transactions
# ============================================================================
print("\n=== Step 7: Scoring Transactions ===")

# Prepare scoring data
score_feature_cols = [col for col in score_fe.columns if col in feature_cols]
X_score = score_fe[score_feature_cols]

# Fill any missing columns with 0
for col in feature_cols:
    if col not in X_score.columns:
        X_score[col] = 0

# Ensure column order matches training
X_score = X_score[feature_cols]

# Predict
print(f"Scoring {len(X_score)} transactions...")
score_predictions = model.predict_proba(X_score)[:, 1]

# Create predictions DataFrame
predictions_df = pd.DataFrame({
    'transaction_id': score_df['transaction_id'],
    'fraud_probability': score_predictions,
})

print(f"Predictions range: [{predictions_df['fraud_probability'].min():.4f}, {predictions_df['fraud_probability'].max():.4f}]")
print(f"Predictions mean: {predictions_df['fraud_probability'].mean():.4f}")

# ============================================================================
# Step 8: Create Predictions Feature Table
# ============================================================================
print("\n=== Step 8: Creating Predictions Feature Table ===")

pred_fg_name = "ccpredee3558"

# Check if predictions feature group already exists
try:
    pred_fg = fs.get_feature_group(pred_fg_name, version=1)
    print(f"Predictions feature group {pred_fg_name} already exists")
except:
    from hsfs import feature
    
    # Create predictions feature group
    pred_features = [
        feature.Feature("transaction_id", "string", primary=True),
        feature.Feature("fraud_probability", "float"),
    ]
    
    pred_fg = fs.create_feature_group(
        name=pred_fg_name,
        version=1,
        description="Credit card fraud detection predictions",
        primary_key=["transaction_id"],
        online_enabled=True,
        features=pred_features,
    )
    print(f"Created predictions feature group: {pred_fg_name}")

# Insert predictions
print(f"Inserting {len(predictions_df)} predictions...")
pred_fg.insert(predictions_df, overwrite=True, wait=True)
print("Predictions inserted successfully")

print("\n=== Job Complete ===")
print(f"Feature group: {fg_name}")
print(f"Training dataset: {td_name}")
print(f"Model: {model_name} (ROC AUC: {roc_auc:.4f})")
print(f"Predictions feature table: {pred_fg_name}")
