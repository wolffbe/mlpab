import hopsworks
from hopsworks_common.project import Project
import pandas as pd
import numpy as np
import joblib
import os

# Connect to Hopsworks
hopsworks.login()
project = Project()
fs = project.get_feature_store()

# Load the trained model
model_path = "/hopsfs/Resources/model/ccmodelee3558.pkl"
model = joblib.load(model_path)
print(f"Model loaded from {model_path}")

# Read score transactions
df_score = pd.read_csv("/hopsfs/Resources/score_transactions.csv")
print(f"Score transactions shape: {df_score.shape}")

# Feature engineering for score transactions
# We need to apply the same feature engineering as for the training data

# 1. Parse datetime
df_score['datetime'] = pd.to_datetime(df_score['datetime'])
df_score['datetime'] = df_score['datetime'].astype('int64') // 10**6  # Convert to milliseconds

# 2. Calculate time-based features
df_score['hour_of_day'] = pd.to_datetime(df_score['datetime'] * 10**6).dt.hour
df_score['day_of_week'] = pd.to_datetime(df_score['datetime'] * 10**6).dt.dayofweek
df_score['day_of_month'] = pd.to_datetime(df_score['datetime'] * 10**6).dt.day

# 3. Calculate transaction velocity per card
score_card_transactions = df_score.groupby('cc_num')

def calculate_velocity(group):
    group = group.sort_values('datetime')
    group['time_since_last_txn'] = group['datetime'].diff() / 3600000  # in hours (datetime is in ms)
    group['txn_velocity'] = 1.0 / (group['time_since_last_txn'] + 1e-6)
    return group

df_score = score_card_transactions.apply(calculate_velocity).reset_index(drop=True)

# 4. Calculate card's usual location from training data (we need to get this from the feature group)
# Read the training feature group to get card statistics
fg = fs.get_feature_group("cctxnee3558", version=1)
df_train = fg.read()

# Get card's usual location
card_locations = df_train.groupby('cc_num')[['lat', 'long']].mean().reset_index()
card_locations = card_locations.rename(columns={'lat': 'card_mean_lat', 'long': 'card_mean_long'})
df_score = df_score.merge(card_locations, on='cc_num', how='left')

# 5. Calculate geo distance from usual location
def haversine(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def calc_distance(row):
    if pd.isna(row['card_mean_lat']) or pd.isna(row['card_mean_long']):
        return 0.0
    return haversine(row['lat'], row['long'], row['card_mean_lat'], row['card_mean_long'])

df_score['geo_distance_from_usual_km'] = df_score.apply(calc_distance, axis=1)

# 6. Calculate amount features using training data statistics
card_amount_mean = df_train.groupby('cc_num')['amount'].mean().to_dict()
df_score['rolling_amount_mean_5'] = np.nan
df_score['rolling_amount_std_5'] = np.nan
df_score['amount_ratio_to_avg'] = df_score.apply(
    lambda row: row['amount'] / card_amount_mean.get(row['cc_num'], row['amount']),
    axis=1
)

# 7. Category and merchant encoding from training
category_counts = df_train['category'].value_counts(normalize=True).to_dict()
merchant_counts = df_train['merchant'].value_counts(normalize=True).to_dict()

df_score['category_freq'] = df_score['category'].map(category_counts)
df_score['merchant_freq'] = df_score['merchant'].map(merchant_counts)

# 8. Time since first transaction for the card (from training)
card_first_txn = df_train.groupby('cc_num')['datetime'].min().to_dict()
df_score['card_first_txn_time'] = df_score['cc_num'].map(card_first_txn)
df_score['time_since_card_first_txn_hours'] = (df_score['datetime'] - df_score['card_first_txn_time']) / 3600000  # Convert ms to hours

# For cards not in training, use the score transaction time as first txn
df_score['card_first_txn_time'] = df_score['card_first_txn_time'].fillna(df_score['datetime'])
df_score['time_since_card_first_txn_hours'] = df_score['time_since_card_first_txn_hours'].fillna(0.0)

# 9. Transaction count per card up to this point
df_score['card_txn_count'] = df_score.groupby('cc_num').cumcount() + 1

# Select features for prediction (same as training)
feature_columns = [
    'cc_num',
    'datetime',
    'amount',
    'merchant',
    'category',
    'lat',
    'long',
    'hour_of_day',
    'day_of_week',
    'day_of_month',
    'txn_velocity',
    'geo_distance_from_usual_km',
    'rolling_amount_mean_5',
    'rolling_amount_std_5',
    'amount_ratio_to_avg',
    'category_freq',
    'merchant_freq',
    'time_since_card_first_txn_hours',
    'card_txn_count'
]

X_score = df_score[feature_columns].copy()

# Encode categorical columns (same as training)
categorical_cols = ['merchant', 'category']
X_score = pd.get_dummies(X_score, columns=categorical_cols, drop_first=True)

print(f"Score features shape: {X_score.shape}")

# Make predictions
y_pred_proba = model.predict_proba(X_score)[:, 1]

# Create predictions dataframe
predictions = pd.DataFrame({
    'transaction_id': df_score['transaction_id'],
    'fraud_probability': y_pred_proba
})

print(f"Predictions shape: {predictions.shape}")
print(f"Predictions head:\n{predictions.head()}")

# Save predictions
predictions_path = "/hopsfs/Resources/predictions/ccpredee3558.csv"
os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
predictions.to_csv(predictions_path, index=False)
print(f"Predictions saved to {predictions_path}")

# Also create a feature group for the predictions
# First, check if the feature group exists
fg_name = "ccpredee3558"
fg_version = 1

try:
    fg_pred = fs.get_feature_group(fg_name, version=fg_version)
    print(f"Feature group {fg_name} already exists")
except:
    fg_pred = fs.create_feature_group(
        name=fg_name,
        version=fg_version,
        primary_key=['transaction_id'],
        event_time=None,
        online_enabled=True,
        description="Credit card fraud predictions"
    )
    print(f"Created feature group {fg_name}")

# Insert predictions into feature group
fg_pred.insert(predictions, write_options={"wait_for_job": True})
print(f"Inserted {len(predictions)} rows into feature group {fg_name}")

print("Scoring complete!")
