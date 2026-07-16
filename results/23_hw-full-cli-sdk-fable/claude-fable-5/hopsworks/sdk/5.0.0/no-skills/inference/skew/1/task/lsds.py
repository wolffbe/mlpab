import urllib3

urllib3.disable_warnings()
import hopsworks

proj = hopsworks.login()
ds = proj.get_dataset_api()

for p in ["Resources", "Resources/skew_task"]:
    print("==", p, "exists:", ds.exists(p))
    try:
        count, files = ds.list_files(p, 0, 100)
        for f in files:
            print("  ", f.attributes.path, f.attributes.size)
    except Exception as e:
        print("  list_files error:", type(e).__name__, e)

for f in ["Resources/skew_task/training_sample.csv",
          "Resources/skew_task/serving_log.csv",
          "Resources/skew_task/skew_job.py"]:
    print(f, "exists:", ds.exists(f))
