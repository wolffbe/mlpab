import hopsworks
import inspect
from hsfs import feature_view as fv_module

print("Connecting...")
project = hopsworks.login()
fs = project.get_feature_store()

print("\n=== FeatureView class methods ===")
methods = [m for m in dir(fv_module.FeatureView) if not m.startswith('_')]
print(methods)

print("\n=== create_training_data signature ===")
if hasattr(fv_module.FeatureView, 'create_training_data'):
    print(inspect.signature(fv_module.FeatureView.create_training_data))

print("\n=== training_data signature ===")
if hasattr(fv_module.FeatureView, 'training_data'):
    print(inspect.signature(fv_module.FeatureView.training_data))

print("\n=== get_training_data signature ===")
if hasattr(fv_module.FeatureView, 'get_training_data'):
    print(inspect.signature(fv_module.FeatureView.get_training_data))

print("\n=== create_train_test_split ===")
if hasattr(fv_module.FeatureView, 'create_train_test_split'):
    print(inspect.signature(fv_module.FeatureView.create_train_test_split))

# Check Query join methods
print("\n=== Query join signature ===")
from hsfs.constructor import query as query_module
q_methods = [m for m in dir(query_module.Query) if not m.startswith('_')]
print(q_methods)
if hasattr(query_module.Query, 'join'):
    print("\njoin signature:", inspect.signature(query_module.Query.join))

# Check FeatureGroup.select_all
print("\n=== FeatureGroup methods ===")
from hsfs import feature_group as fg_module
fg_methods = [m for m in dir(fg_module.FeatureGroup) if not m.startswith('_')]
print(fg_methods)
if hasattr(fg_module.FeatureGroup, 'select_all'):
    print("\nselect_all signature:", inspect.signature(fg_module.FeatureGroup.select_all))
