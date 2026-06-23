import hopsworks
import pandas as pd
import os
import glob
from datetime import datetime

project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_feature_group(name="incremental811051", version=1)

# Find the next increment file to ingest
# In production, this would look for new files in a specific location
data_dir = "/srv/hops/staging"
increment_files = sorted(glob.glob(os.path.join(data_dir, "increment_*.csv")))

if increment_files:
    for filepath in increment_files:
        df = pd.read_csv(filepath)
        fg.insert(df)
        print(f"Ingested {filepath}: {len(df)} rows at {datetime.utcnow()}")
else:
    print(f"No increment files found in {data_dir}")
