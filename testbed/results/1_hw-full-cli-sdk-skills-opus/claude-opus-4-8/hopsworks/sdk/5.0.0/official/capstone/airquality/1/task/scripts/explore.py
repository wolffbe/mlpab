import warnings
warnings.filterwarnings("ignore")
import hopsworks

project = hopsworks.login()
print("PROJECT:", project.name)
fs = project.get_feature_store()
print("FS:", fs.name)

ja = project.get_jobs_api()
print("JOBS API:", [m for m in dir(ja) if not m.startswith("_")])

ds = project.get_dataset_api()
print("DATASET API:", [m for m in dir(ds) if not m.startswith("_")])

try:
    ea = project.get_environment_api()
    print("ENV API:", [m for m in dir(ea) if not m.startswith("_")])
    envs = ea.get_environments()
    print("ENVIRONMENTS:", [e.name for e in envs])
except Exception as e:
    print("env err", repr(e))
