"""Runs on the cluster: verify airqpred754fa9 offline + online read paths (with retries)."""

import os
import time

import hopsworks

lines = []


def emit(*args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    lines.append(msg)


project = hopsworks.login()
fs = project.get_feature_store()

pred_fg = fs.get_feature_group("airqpred754fa9", 1)

df = None
for attempt in range(6):
    try:
        df = pred_fg.read()
        break
    except Exception as e:  # noqa: BLE001
        emit(f"offline read attempt {attempt} failed:", e)
        time.sleep(20)

if df is not None:
    emit("OFFLINE ROWS:", len(df))
    emit(df.sort_values("date").head(5).to_string())
    emit("null preds:", int(df["pm25_pred"].isna().sum()))
else:
    emit("OFFLINE READ FAILED AFTER RETRIES")

try:
    df_on = pred_fg.read(online=True)
    emit("ONLINE ROWS:", len(df_on))
    emit(df_on.sort_values("date").head(3).to_string())
except Exception as e:  # noqa: BLE001
    emit("online read failed:", e)

emit("VERIFY_DONE")

out_path = "/hopsfs/Resources/airq754fa9/verify_result.txt"
try:
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("result written to", out_path, flush=True)
except Exception as e:  # noqa: BLE001
    print("result write failed:", e, flush=True)
    dataset_api = project.get_dataset_api()
    with open("verify_result.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    dataset_api.upload("verify_result.txt", "Resources/airq754fa9", overwrite=True)
