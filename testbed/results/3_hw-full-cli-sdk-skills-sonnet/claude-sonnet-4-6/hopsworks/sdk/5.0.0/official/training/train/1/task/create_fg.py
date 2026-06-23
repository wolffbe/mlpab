"""Download predictions.csv, create feature group, and load predictions."""
import os
import pandas as pd
import hopsworks

JOB_NAME = "trainjob7a8727"
FG_NAME = "predictions7a8727"
FG_VERSION = 1
PREDICTIONS_PATH = f"Resources/{JOB_NAME}/predictions.csv"
LOCAL_PREDICTIONS = "/tmp/predictions7a8727.csv"


def main():
    project = hopsworks.login()
    print(f"Connected to project: {project.name}")

    dataset_api = project.get_dataset_api()

    # Download predictions.csv from HopsFS
    print(f"Downloading {PREDICTIONS_PATH}...")
    dataset_api.download(PREDICTIONS_PATH, local_path=LOCAL_PREDICTIONS, overwrite=True)
    print(f"Downloaded to {LOCAL_PREDICTIONS}")

    # Read predictions
    df = pd.read_csv(LOCAL_PREDICTIONS)
    print(f"Predictions shape: {df.shape}")
    print(f"Predictions columns: {df.columns.tolist()}")
    print(df.head())

    # Get feature store
    fs = project.get_feature_store()

    # Create feature group with online enabled
    print(f"\nCreating feature group '{FG_NAME}' v{FG_VERSION}...")
    fg = fs.create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        primary_key=["row_id"],
        online_enabled=True,
        description="Predictions from training job trainjob7a8727",
    )
    print(f"Feature group created: {fg.name} v{fg.version}")

    # Insert predictions
    print("Inserting predictions into feature group...")
    fg.insert(df)
    print("Predictions inserted successfully.")

    print(f"\nDone! Feature group '{FG_NAME}' v{FG_VERSION} is ready with online access enabled.")


if __name__ == "__main__":
    main()
