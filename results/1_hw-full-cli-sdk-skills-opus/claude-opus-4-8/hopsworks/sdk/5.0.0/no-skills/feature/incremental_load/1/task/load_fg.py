import glob
import pandas as pd
import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()

fg = fs.get_or_create_feature_group(
    name="incremental614551",
    version=1,
    description="Daily events increments table",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
)
print("FG:", fg.name, "v", fg.version, "online_enabled=", fg.online_enabled)

files = sorted(glob.glob("data/increment_*.csv"))
print("Files:", files)
total = 0
for f in files:
    df = pd.read_csv(f)
    total += len(df)
    print("Inserting", f, len(df), "rows")
    fg.insert(df, write_options={"wait_for_job": True})
print("TOTAL ROWS INSERTED:", total)
