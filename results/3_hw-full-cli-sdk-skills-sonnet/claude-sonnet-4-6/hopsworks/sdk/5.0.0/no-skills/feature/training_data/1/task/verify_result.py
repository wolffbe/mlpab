import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

FV_NAME = "churntraining2d2b0c"
FV_VERSION = 1

fv = fs.get_feature_view(FV_NAME, version=FV_VERSION)
if fv is None:
    print(f"ERROR: Feature view {FV_NAME} v{FV_VERSION} not found!")
    exit(1)

print(f"Feature view {FV_NAME} v{FV_VERSION} found")
print(f"Labels: {fv.labels}")

# List training datasets
tds = fv.get_training_datasets()
print(f"Training datasets: {[td.version for td in tds] if tds else []}")

# Read back training data version 1
print("\nReading training data v1...")
X, y = fv.get_training_data(training_dataset_version=1)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape if y is not None else 'None'}")
print(f"\nX columns: {list(X.columns)}")
if y is not None:
    print(f"y columns: {list(y.columns)}")

# Combine
if y is not None:
    df = pd.concat([X, y], axis=1)
else:
    df = X

print(f"\nFull DF shape: {df.shape}")
print(f"Full DF columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head(5).to_string())

print(f"\nNull values per column:")
print(df.isnull().sum())
