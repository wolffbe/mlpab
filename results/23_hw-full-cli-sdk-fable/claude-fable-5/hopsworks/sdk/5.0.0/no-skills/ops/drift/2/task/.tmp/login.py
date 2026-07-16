import hw_env  # noqa: F401
import hopsworks

proj = hopsworks.login()
print("project:", proj.name)
fs = proj.get_feature_store()
print("fs:", fs.name)
