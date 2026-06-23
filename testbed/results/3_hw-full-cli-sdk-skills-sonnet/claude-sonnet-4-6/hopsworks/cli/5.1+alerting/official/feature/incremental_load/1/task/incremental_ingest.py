import os
import glob
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("incremental811051", version=1)

data_dir = os.environ.get("HOPS_DATA_DIR", ".")
pattern = os.path.join(data_dir, "increment_*.csv")
files = sorted(glob.glob(pattern))

if files:
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fg.insert(df)
