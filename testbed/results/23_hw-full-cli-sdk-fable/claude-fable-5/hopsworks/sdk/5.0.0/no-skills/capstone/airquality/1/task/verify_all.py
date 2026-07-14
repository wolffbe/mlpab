import time

import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_feature_group("airq754fa9", 1)
print("FG airq754fa9:", fg.id, "features:", [f.name for f in fg.features])

fv = fs.get_feature_view("airqtd754fa9", 1)
print("FV airqtd754fa9:", fv.version, "labels:", fv.labels)

mr = project.get_model_registry()
model = mr.get_model("airqmodel754fa9", 1)
print("MODEL airqmodel754fa9 v1 metrics:", model.training_metrics)

pred_fg = fs.get_feature_group("airqpred754fa9", 1)
print("PRED FG online_enabled:", pred_fg.online_enabled)

# offline read (may lag behind materialization job)
for attempt in range(20):
    try:
        df = pred_fg.read()
        print("offline rows:", len(df))
        if len(df) >= 90:
            print(df.sort_values("date").head(5).to_string())
            break
    except Exception as e:  # noqa: BLE001
        print("offline read failed:", e)
    time.sleep(30)

# online read
try:
    df_on = pred_fg.read(online=True)
    print("online rows:", len(df_on))
    print(df_on.sort_values("date").head(5).to_string())
except Exception as e:  # noqa: BLE001
    print("online read failed:", e)
