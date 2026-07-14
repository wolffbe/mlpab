import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

df = pd.read_csv("data/training_data.csv")
print(df.shape)

fg = fs.get_or_create_feature_group(
    name="leakage_training_data",
    version=1,
    description="Training data for leakage detection task",
    primary_key=["row_id"],
    online_enabled=False,
)
fg.insert(df, wait=True)
print("inserted")
