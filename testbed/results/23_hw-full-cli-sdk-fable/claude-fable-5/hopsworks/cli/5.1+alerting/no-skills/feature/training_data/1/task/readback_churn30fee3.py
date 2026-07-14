"""Platform job: read training dataset churntraining30fee3 v1 via the FV API (REST fallback)."""
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fv = fs.get_feature_view("churntraining30fee3", 1)
X, y = fv.get_training_data(1, read_options={"use_spark": True})
print("X COLUMNS:", list(X.columns))
print("Y COLUMNS:", list(y.columns))
print("ROWS:", len(X))
print(X.head(3).to_string())
print(y.head(3).to_string())
