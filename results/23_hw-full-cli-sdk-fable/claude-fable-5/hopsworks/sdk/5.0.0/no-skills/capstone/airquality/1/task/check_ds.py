import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
for p in [
    "Resources/airq754fa9",
    "Resources/airq754fa9/airquality_history.csv",
    "Resources/airq754fa9/forecast_days.csv",
    "Resources/airq754fa9/pipeline_job.py",
]:
    try:
        print(p, "exists:", ds.exists(p))
    except Exception as e:  # noqa: BLE001
        print(p, "error:", e)

try:
    listing = ds.list("Resources/airq754fa9")
    items = listing.get("items", listing)
    if isinstance(items, list):
        for it in items:
            attr = it.get("attributes", {}) if isinstance(it, dict) else it
            print("ITEM:", attr.get("path", attr), attr.get("size"))
    else:
        print(listing)
except Exception as e:  # noqa: BLE001
    print("list error:", e)
