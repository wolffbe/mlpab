import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()

for p in [
    "Resources",
    "Resources/featureseb4964",
    "Resources/featureseb4964/transactions.csv",
    "Resources/featureseb4964/fg_job.py",
]:
    try:
        print(p, "exists:", ds.exists(p))
    except Exception as e:
        print(p, "error:", e)

res = ds.list("Resources/featureseb4964")
try:
    items = res["items"]
except TypeError:
    items = res
print(type(res))
print(res if isinstance(res, (list, tuple)) else str(res)[:2000])
