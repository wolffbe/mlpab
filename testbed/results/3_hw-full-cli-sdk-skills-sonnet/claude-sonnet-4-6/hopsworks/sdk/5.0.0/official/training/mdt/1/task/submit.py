import hopsworks
import pandas as pd
import math

# Read data
train_df = pd.read_csv("data/features_train.csv")
serve_df = pd.read_csv("data/features_serve.csv")

features = ["f1", "f2", "f3", "f4"]

# Compute training stats (population std - no Bessel's correction)
means = {}
stds = {}
for f in features:
    col = train_df[f]
    means[f] = col.mean()
    # Population std
    stds[f] = col.std(ddof=0)

print("Training stats:")
for f in features:
    print(f"  {f}: mean={means[f]}, std={stds[f]}")

# Standardize both splits using training stats
def standardize(df, split_name):
    result = df.copy()
    result["split"] = split_name
    for f in features:
        result[f] = ((df[f] - means[f]) / stds[f]).round(6)
    return result

train_std = standardize(train_df, "train")
serve_std = standardize(serve_df, "serve")

# Combine and select columns in correct order
combined = pd.concat([train_std, serve_std], ignore_index=True)
combined = combined[["row_id", "split", "f1", "f2", "f3", "f4"]]

print(f"Combined shape: {combined.shape}")
print(combined.head())

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Create the feature group with online enabled
fg = fs.get_or_create_feature_group(
    name="scaled560eee",
    version=1,
    primary_key=["row_id"],
    online_enabled=True,
    description="Standardized features using training split statistics"
)

print("Feature group created/retrieved:", fg.name, "v", fg.version)

# Insert data
fg.insert(combined, write_options={"wait_for_job": True})
print("Data inserted successfully")
