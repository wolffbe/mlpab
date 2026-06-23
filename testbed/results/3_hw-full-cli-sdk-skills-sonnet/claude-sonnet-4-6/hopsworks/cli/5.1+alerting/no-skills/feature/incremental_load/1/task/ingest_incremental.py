import os
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("incremental811051", version=1)

file_path = os.environ.get("HOPS_FILE_PATH", "")
if file_path:
    import pandas as pd
    df = pd.read_csv(file_path)
    fg.insert(df)
