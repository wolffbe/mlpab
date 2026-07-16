import hopsworks
import pandas as pd
import os

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()
dataset_api.download("Resources/features_train.csv", local_path="/tmp/features_train.csv", overwrite=True)
dataset_api.download("Resources/features_serve.csv", local_path="/tmp/features_serve.csv", overwrite=True)

train_df = pd.read_csv("/tmp/features_train.csv")
serve_df = pd.read_csv("/tmp/features_serve.csv")

features = ["f1", "f2", "f3", "f4"]
means = train_df[features].mean()
stds = train_df[features].std(ddof=0)

train_scaled = train_df.copy()
serve_scaled = serve_df.copy()
for f in features:
    train_scaled[f] = ((train_df[f] - means[f]) / stds[f]).round(6)
    serve_scaled[f] = ((serve_df[f] - means[f]) / stds[f]).round(6)

train_scaled["split"] = "train"
serve_scaled["split"] = "serve"

combined = pd.concat([train_scaled[["row_id", "split"] + features],
                      serve_scaled[["row_id", "split"] + features]],
                     ignore_index=True)

fg = fs.get_or_create_feature_group(
    name="scaled560eee",
    version=1,
    primary_key=["row_id"],
    online_enabled=True,
    description="Standardized features (mean/std from training split)"
)

fg.insert(combined, wait=True)
print("Done. Rows inserted:", len(combined))
