import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
res = ds.list("Resources/transactions_ingest")
print(type(res))
print(res)
for p in (
    "Resources/transactions_ingest/ingest_job.py",
    "Resources/transactions_ingest/transactions_export_1.csv",
    "Resources/transactions_ingest/transactions_export_2.csv",
):
    print(p, "exists:", ds.exists(p))
