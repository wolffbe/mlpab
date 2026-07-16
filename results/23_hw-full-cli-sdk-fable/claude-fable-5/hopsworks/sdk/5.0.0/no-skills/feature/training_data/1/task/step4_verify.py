import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

fv = fs.get_feature_view("churntraining30fee3", 1)
print("feature view:", fv.name, "version:", fv.version)
print("labels:", fv.labels)

try:
    X, y = fv.get_training_data(1)
    print("X shape:", X.shape, "y shape:", y.shape)
    print("X columns:", list(X.columns))
    print("y columns:", list(y.columns))
    print(X.head(3))
    print(y.head(3))
    print("nulls per column:")
    print(X.isna().sum())
except Exception as e:
    print("get_training_data failed:", type(e).__name__, e)
