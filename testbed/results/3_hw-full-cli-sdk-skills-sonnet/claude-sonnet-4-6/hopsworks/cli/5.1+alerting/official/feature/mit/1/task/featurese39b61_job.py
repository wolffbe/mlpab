import hopsworks
import pandas as pd
import os

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()
dataset_api.download("Resources/featurese39b61_data/transactions.csv", local_path="/tmp/transactions.csv", overwrite=True)
dataset_api.download("Resources/featurese39b61_data/fx_rates.csv", local_path="/tmp/fx_rates.csv", overwrite=True)

transactions = pd.read_csv("/tmp/transactions.csv")
fx_rates = pd.read_csv("/tmp/fx_rates.csv")

df = transactions.merge(fx_rates, on='currency', how='left')

df['amount_usd'] = df['amount'] * df['fx_rate']

df['event_time_dt'] = pd.to_datetime(df['event_time'], unit='ms', utc=True)
df['is_weekend'] = df['event_time_dt'].dt.dayofweek.isin([5, 6]).astype(int)

SEVEN_DAYS_MS = 7 * 24 * 3600 * 1000
df_left = df[['row_id', 'account_id', 'event_time', 'amount']].copy()
df_right = df[['account_id', 'event_time', 'amount']].copy()
df_left['window_start'] = df_left['event_time'] - SEVEN_DAYS_MS

merged = pd.merge(df_left, df_right, on='account_id', suffixes=('', '_r'))
merged = merged[
    (merged['event_time_r'] >= merged['window_start']) &
    (merged['event_time_r'] <= merged['event_time'])
]
amount_7d = merged.groupby('row_id')['amount_r'].sum().reset_index()
amount_7d.columns = ['row_id', 'amount_7d']

df = df.merge(amount_7d, on='row_id', how='left')

result = df[['row_id', 'account_id', 'event_time', 'amount_usd', 'is_weekend', 'amount_7d']].copy()
result['is_weekend'] = result['is_weekend'].astype(int)
result['event_time'] = result['event_time'].astype('int64')

print("Sample rows:")
print(result.head())
print("Shape:", result.shape)

fg = fs.get_or_create_feature_group(
    name="featurese39b61",
    version=1,
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
    description="Derived transaction features with amount_usd, is_weekend, amount_7d"
)

fg.insert(result, wait=True)
print("Insert complete.")
