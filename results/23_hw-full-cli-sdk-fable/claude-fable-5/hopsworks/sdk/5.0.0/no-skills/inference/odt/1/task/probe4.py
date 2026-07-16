import hopsworks, inspect
project = hopsworks.login()
print("project:", project.name)
fs = project.get_feature_store()
print("fs:", fs.name)
print(inspect.signature(fs.create_feature_group))
import importlib
fgmod = importlib.import_module("hsfs.feature_group")
FG = fgmod.FeatureGroup
print("insert sig:", inspect.signature(FG.insert))
