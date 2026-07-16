import hopsworks
import inspect

print("Connecting...")
project = hopsworks.login()
fs = project.get_feature_store()

print("\n=== Feature Store methods ===")
methods = [m for m in dir(fs) if not m.startswith('_')]
print(methods)

print("\n=== create_training_dataset signature ===")
if hasattr(fs, 'create_training_dataset'):
    print(inspect.signature(fs.create_training_dataset))

print("\n=== create_feature_view signature ===")
if hasattr(fs, 'create_feature_view'):
    print(inspect.signature(fs.create_feature_view))

print("\n=== FeatureView methods ===")
# Try to get a feature view if it exists
try:
    fvs = fs.get_feature_views()
    if fvs:
        fv = fvs[0]
        fv_methods = [m for m in dir(fv) if not m.startswith('_')]
        print(fv_methods)

        print("\n=== create_training_data signature ===")
        if hasattr(fv, 'create_training_data'):
            print(inspect.signature(fv.create_training_data))

        print("\n=== training_data signature ===")
        if hasattr(fv, 'training_data'):
            print(inspect.signature(fv.training_data))

        print("\n=== get_training_data signature ===")
        if hasattr(fv, 'get_training_data'):
            print(inspect.signature(fv.get_training_data))
except Exception as e:
    print(f"No feature views yet: {e}")
