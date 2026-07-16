import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()

path = ds.upload(
    "data/transactions_export_1.csv",
    "Resources/transactions_ingest",
    overwrite=True,
)
print("uploaded:", path)
print("exists:", ds.exists("Resources/transactions_ingest/transactions_export_1.csv"))
print(ds.list("Resources/transactions_ingest"))
