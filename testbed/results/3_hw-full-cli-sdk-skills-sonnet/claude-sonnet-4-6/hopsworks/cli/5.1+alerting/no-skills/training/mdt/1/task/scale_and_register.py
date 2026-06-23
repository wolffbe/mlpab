import hopsworks
import pandas as pd
import numpy as np
import os

def main():
    project = hopsworks.login()
    fs = project.get_feature_store()
    dataset_api = project.get_dataset_api()

    dataset_api.download("Resources/features_train.csv", local_path="/tmp/features_train.csv", overwrite=True)
    dataset_api.download("Resources/features_serve.csv", local_path="/tmp/features_serve.csv", overwrite=True)

    train_df = pd.read_csv("/tmp/features_train.csv")
    serve_df = pd.read_csv("/tmp/features_serve.csv")

    features = ['f1', 'f2', 'f3', 'f4']
    means = train_df[features].mean()
    stds = train_df[features].std(ddof=0)

    train_std = train_df.copy()
    serve_std = serve_df.copy()

    for f in features:
        train_std[f] = ((train_df[f] - means[f]) / stds[f]).round(6)
        serve_std[f] = ((serve_df[f] - means[f]) / stds[f]).round(6)

    train_std['split'] = 'train'
    serve_std['split'] = 'serve'

    combined = pd.concat([train_std, serve_std], ignore_index=True)
    combined = combined[['row_id', 'split', 'f1', 'f2', 'f3', 'f4']]

    fg = fs.create_feature_group(
        name="scaled560eee",
        version=1,
        primary_key=["row_id"],
        online_enabled=True,
        description="Standardized feature table with train and serve splits",
    )

    fg.insert(combined)
    print("Feature group registered and data inserted successfully.")

if __name__ == "__main__":
    main()
