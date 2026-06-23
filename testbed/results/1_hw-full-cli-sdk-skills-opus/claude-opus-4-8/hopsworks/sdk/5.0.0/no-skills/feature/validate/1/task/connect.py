import warnings, urllib3
warnings.filterwarnings("ignore")
urllib3.disable_warnings()
import hopsworks
proj = hopsworks.login()
print("PROJECT:", proj.name)
fs = proj.get_feature_store()
print("FS:", fs.name)
import inspect
print(inspect.signature(fs.create_feature_group))
