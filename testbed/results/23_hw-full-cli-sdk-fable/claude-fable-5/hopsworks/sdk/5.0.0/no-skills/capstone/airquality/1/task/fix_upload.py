import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
for attempt in range(5):
    try:
        ds.upload("data/airquality_history.csv", "Resources/airq754fa9", overwrite=True)
    except Exception as e:  # noqa: BLE001
        print("upload attempt failed:", e)
        continue
    if ds.exists("Resources/airq754fa9/airquality_history.csv"):
        print("upload verified")
        break
    print("still missing, retrying")
else:
    raise SystemExit("upload failed after retries")
