import hopsworks
import pandas as pd
import numpy as np
from datetime import datetime

# Connect to Hopsworks
import hopsworks
from hopsworks_common.project import Project

try:
    hopsworks.login()
    project = Project()
    fs = project.get_feature_store()
    print("Connected to Hopsworks")
except Exception as e:
    print(f"Error connecting to Hopsworks: {e}")
    import traceback
    traceback.print_exc()
    raise

# Read the transactions data from HopsFS
import os

# Read data directly from /hopsfs/Resources/
df = pd.read_csv("/hopsfs/Resources/transactions.csv")
df_score = pd.read_csv("/hopsfs/Resources/score_transactions.csv")

print(f"Transactions shape: {df.shape}")
print(f"Score transactions shape: {df_score.shape}")

# Feature Engineering

# 1. Parse datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df_score['datetime'] = pd.to_datetime(df_score['datetime'])

# 2. Calculate time-based features
df['hour_of_day'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['day_of_month'] = df['datetime'].dt.day

df_score['hour_of_day'] = df_score['datetime'].dt.hour
df_score['day_of_week'] = df_score['datetime'].dt.dayofweek
df_score['day_of_month'] = df_score['datetime'].dt.day

# 3. Calculate transaction velocity per card (transactions per hour)
# For each card, calculate time between consecutive transactions
card_transactions = df.groupby('cc_num')

def calculate_velocity(group):
    group = group.sort_values('datetime')
    group['time_since_last_txn'] = group['datetime'].diff().dt.total_seconds() / 3600  # in hours
    group['txn_velocity'] = 1.0 / (group['time_since_last_txn'] + 1e-6)  # transactions per hour
    return group

df = card_transactions.apply(calculate_velocity).reset_index(drop=True)

# For score data
score_card_transactions = df_score.groupby('cc_num')
score_df_temp = score_card_transactions.apply(calculate_velocity).reset_index(drop=True)
df_score = score_df_temp

# 4. Calculate card's usual location (mean lat/long per card) from training data
card_locations = df.groupby('cc_num')[['lat', 'long']].mean().reset_index()
card_locations = card_locations.rename(columns={'lat': 'card_mean_lat', 'long': 'card_mean_long'})

# Merge back to get usual location for each transaction
df = df.merge(card_locations, on='cc_num', how='left')
df_score = df_score.merge(card_locations, on='cc_num', how='left')

# 5. Calculate geo distance from usual location (in km)
# Simple haversine formula
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

df['geo_distance_from_usual_km'] = df.apply(calc_distance, axis=1)
df_score['geo_distance_from_usual_km'] = df_score.apply(calc_distance, axis=1)

# 6. Calculate amount features
# Rolling mean of transaction amounts per card
df['rolling_amount_mean_5'] = card_transactions['amount'].transform(
    lambda x: x.rolling(5, min_periods=1).mean()
)
df['rolling_amount_std_5'] = card_transactions['amount'].transform(
    lambda x: x.rolling(5, min_periods=1).std()
)

# For score data, use card-level statistics from training
df_score['rolling_amount_mean_5'] = np.nan
df_score['rolling_amount_std_5'] = np.nan

# Amount relative to card's average
df['amount_ratio_to_avg'] = df['amount'] / df.groupby('cc_num')['amount'].transform('mean')

# For score, use training data statistics
card_amount_mean = df.groupby('cc_num')['amount'].mean().to_dict()
df_score['amount_ratio_to_avg'] = df_score.apply(
    lambda row: row['amount'] / card_amount_mean.get(row['cc_num'], row['amount']),
    axis=1
)

# 7. Category and merchant encoding (simple frequency encoding)
category_counts = df['category'].value_counts(normalize=True).to_dict()
merchant_counts = df['merchant'].value_counts(normalize=True).to_dict()

df['category_freq'] = df['category'].map(category_counts)
df['merchant_freq'] = df['merchant'].map(merchant_counts)

df_score['category_freq'] = df_score['category'].map(category_counts)
df_score['merchant_freq'] = df_score['merchant'].map(merchant_counts)

# 8. Time since first transaction for the card
df['card_first_txn_time'] = df.groupby('cc_num')['datetime'].transform('min')
df['time_since_card_first_txn_hours'] = (df['datetime'] - df['card_first_txn_time']).dt.total_seconds() / 3600

# For score data
card_first_txn = df.groupby('cc_num')['datetime'].min().to_dict()
df_score['card_first_txn_time'] = df_score['cc_num'].map(card_first_txn)
df_score['time_since_card_first_txn_hours'] = (df_score['datetime'] - df_score['card_first_txn_time']).dt.total_seconds() / 3600

# For cards not in training, use the score transaction time as first txn
df_score['card_first_txn_time'] = df_score['card_first_txn_time'].fillna(df_score['datetime'])
df_score['time_since_card_first_txn_hours'] = df_score['time_since_card_first_txn_hours'].fillna(0.0)

# 9. Transaction count per card up to this point
df['card_txn_count'] = df.groupby('cc_num').cumcount() + 1
df_score['card_txn_count'] = df_score.groupby('cc_num').cumcount() + 1

# Select features for the feature group
feature_columns = [
    'transaction_id',
    'cc_num',
    'datetime',
    'amount',
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
    'card_txn_count',
    'is_fraud'
]

# Filter to only include columns we want
df_features = df[feature_columns].copy()

# Convert types to match feature group schema
df_features['cc_num'] = df_features['cc_num'].astype(str)
df_features['is_fraud'] = df_features['is_fraud'].astype('int32')

print(f"Feature engineered dataframe shape: {df_features.shape}")
print(f"Columns: {df_features.columns.tolist()}")
print(f"cc_num dtype: {df_features['cc_num'].dtype}")
print(f"is_fraud dtype: {df_features['is_fraud'].dtype}")

# Create feature group
fg_name = "cctxnee3558"
fg_version = 1

# Check if feature group exists
try:
    fg = fs.get_feature_group(fg_name, version=fg_version)
    print(f"Feature group {fg_name} already exists")
except:
    # Create new feature group
    fg = fs.create_feature_group(
        name=fg_name,
        version=fg_version,
        primary_key=['transaction_id'],
        event_time='datetime',
        partition_key=['cc_num'],
        online_enabled=True,
        description="Credit card fraud detection features"
    )
    print(f"Created feature group {fg_name}")

# Insert data into feature group
fg.insert(df_features, write_options={"wait_for_job": True})
print(f"Inserted {len(df_features)} rows into feature group {fg_name}")

# Save engineered score features for later use
score_feature_columns = [c for c in feature_columns if c != 'is_fraud']
score_features = df_score[score_feature_columns].copy()
score_features.to_csv("score_features.csv", index=False)

print(f"Score features shape: {score_features.shape}")
print("Feature engineering complete!")
