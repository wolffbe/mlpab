import time

import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("transactions82e347", version=1)
print("online_enabled:", fg.online_enabled, flush=True)

online_count = -1
for attempt in range(20):
    df = fg.read(online=True)
    online_count = len(df)
    print("attempt", attempt, "ONLINE_COUNT:", online_count, flush=True)
    if online_count >= 600:
        break
    time.sleep(15)

df = fg.read(online=True)
print("ONLINE_COUNT_FINAL:", len(df), flush=True)
print("ONLINE_UNIQUE_ROW_IDS:", df["row_id"].nunique(), flush=True)
print(df.sort_values("row_id").head(3).to_string(), flush=True)
print("VERIFY_JOB_DONE", flush=True)
