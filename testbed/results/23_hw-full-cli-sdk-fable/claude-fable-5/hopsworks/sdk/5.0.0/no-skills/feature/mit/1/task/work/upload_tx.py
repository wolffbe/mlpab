import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()

p = ds.upload("data/transactions.csv", "Resources/featureseb4964", overwrite=True)
print("uploaded:", p)
print("exists:", ds.exists("Resources/featureseb4964/transactions.csv"))
print(ds.list("Resources/featureseb4964"))
