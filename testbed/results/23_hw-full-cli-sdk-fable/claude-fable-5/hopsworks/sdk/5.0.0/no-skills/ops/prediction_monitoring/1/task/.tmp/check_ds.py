import hopsworks

proj = hopsworks.login()
dataset_api = proj.get_dataset_api()
for p in [
    "Resources/prediction_log.csv",
    "Resources/remote_monitor_job.py",
    f"/Projects/{proj.name}/Resources/prediction_log.csv",
]:
    try:
        print(p, "exists:", dataset_api.exists(p))
    except Exception as e:
        print(p, "err:", e)

try:
    entries = dataset_api.list_files("Resources", 0, 100)
    print(entries)
except Exception as e:
    print("list_files err:", e)
    import inspect
    print([m for m in dir(dataset_api) if not m.startswith("_")])
