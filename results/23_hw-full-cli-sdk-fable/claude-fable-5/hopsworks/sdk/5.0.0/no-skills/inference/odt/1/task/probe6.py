import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
for p in ["Resources/requests.csv", "Resources/profiles.csv", "Resources/scored_job.py", "Resources"]:
    try:
        print(p, "exists:", ds.exists(p))
    except Exception as e:
        print(p, "err:", e)
try:
    print(ds.list_files("Resources", 0, 100))
except Exception as e:
    print("list_files err:", e)
    print([m for m in dir(ds) if not m.startswith('_')])
