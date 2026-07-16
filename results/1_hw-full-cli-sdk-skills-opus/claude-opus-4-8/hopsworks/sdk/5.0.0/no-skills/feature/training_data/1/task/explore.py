import warnings, urllib3, inspect
warnings.filterwarnings("ignore")
urllib3.disable_warnings()
import hopsworks
proj = hopsworks.login()
fs = proj.get_feature_store()

from hsfs.feature_group import FeatureGroup
print("=== fg.insert sig ===")
print(inspect.signature(FeatureGroup.insert))

from hsfs.feature_view import FeatureView
print("=== fv methods ===")
print([m for m in dir(FeatureView) if not m.startswith('_')])
print("=== fv.create_training_data sig ===")
print(inspect.signature(FeatureView.create_training_data))
print("=== fv.training_data sig ===")
print(inspect.signature(FeatureView.training_data))
print("=== fv.get_training_data sig ===")
print(inspect.signature(FeatureView.get_training_data))
