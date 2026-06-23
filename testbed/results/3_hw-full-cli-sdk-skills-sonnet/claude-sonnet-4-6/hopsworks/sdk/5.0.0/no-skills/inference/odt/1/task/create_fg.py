import hopsworks
import pandas as pd
import math

# Read input data
requests = pd.read_csv("data/requests.csv")
profiles = pd.read_csv("data/profiles.csv")

# Join on account_id
df = requests.merge(profiles, on="account_id", how="inner")

# Compute on-demand features
def compute_distance(row):
    return round(math.sqrt((row["request_lat"] - row["home_lat"])**2 + (row["request_lon"] - row["home_lon"])**2), 6)

df["distance_deg"] = df.apply(compute_distance, axis=1)
df["score"] = (df["base_score"] - 0.1 * df["distance_deg"]).round(6)

# Keep only required columns
result = df[["request_id", "account_id", "distance_deg", "score"]].copy()

print(f"Computed {len(result)} rows")
print(result.head())

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Create feature group with online storage enabled
fg = fs.get_or_create_feature_group(
    name="scoredd87bfb",
    version=1,
    primary_key=["request_id"],
    online_enabled=True,
    description="Score feature group with distance_deg and score",
)

print("Inserting data...")
fg.insert(result, write_options={"wait_for_job": True})
print("Done!")
