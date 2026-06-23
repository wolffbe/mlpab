import hopsworks, warnings
warnings.filterwarnings("ignore")
project = hopsworks.login()
print("Project:", project.name)
fs = project.get_feature_store()
print("FS:", fs.name)
