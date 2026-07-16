import hopsworks
import pandas as pd
import hsfs
from hsfs.statistics_config import StatisticsConfig

print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_feature_group("prediction_log", version=1)
print(f"Feature group: {fg.name} v{fg.version}")

# Read data from the feature group (platform offline store)
print("\nReading data from feature group offline store...")
df = fg.read(dataframe_type="pandas")
print(f"Read {len(df)} rows from feature group")
print(f"Columns: {df.columns.tolist()}")
print(f"dtypes: {df.dtypes}")
print(df.head())

# Compute daily statistics from platform data
df['ts'] = pd.to_datetime(df['ts'])
df['date'] = df['ts'].dt.strftime('%Y-%m-%d')
daily_means = df.groupby('date')['prediction'].agg(['mean', 'std', 'count'])
print("\n=== Daily means from platform data ===")
print(daily_means.to_string())
