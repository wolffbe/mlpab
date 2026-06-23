import warnings
warnings.filterwarnings("ignore")
import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
ds.mkdir("Resources/airq")
up = ds.upload("data/forecast_days.csv", "Resources/airq", overwrite=True)
print("uploaded forecast to:", up)
print("exists:", ds.path_exists("Resources/airq/forecast_days.csv"))
