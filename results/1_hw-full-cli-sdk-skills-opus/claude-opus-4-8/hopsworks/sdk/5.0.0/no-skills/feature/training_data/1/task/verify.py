import warnings, urllib3
warnings.filterwarnings("ignore")
urllib3.disable_warnings()
import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()
fv = fs.get_feature_view("churntrainingc8f821", version=1)

X, y = fv.get_training_data(training_dataset_version=1)
print("READBACK TYPE:", type(X), type(y))
print("COLUMNS:", list(X.columns))
print("SHAPE:", X.shape)
print("UNIQUE accounts:", X["account_id"].nunique())
print("DTYPES:\n", X.dtypes)
print(X.sort_values("account_id").head(5).to_string())
