import hopsworks
import inspect

print("Connecting...")
project = hopsworks.login()
fs = project.get_feature_store()

print("\n=== Hopsworks version ===")
print(hopsworks.__version__)

import hsfs
print("hsfs version:", hsfs.__version__)

print("\n=== get_training_dataset signature ===")
print(inspect.signature(fs.get_training_dataset))

print("\n=== get_training_datasets signature ===")
print(inspect.signature(fs.get_training_datasets))

# Try to get existing training datasets
print("\n=== Existing training datasets ===")
try:
    tds = fs.get_training_datasets()
    print("Training datasets:", tds)
except Exception as e:
    print(f"Error: {e}")

# Check if feature view training datasets are accessible via fs.get_training_dataset
print("\n=== FeatureView get_training_datasets ===")
from hsfs import feature_view as fv_module
if hasattr(fv_module.FeatureView, 'get_training_datasets'):
    print(inspect.signature(fv_module.FeatureView.get_training_datasets))

# Check what schema a training dataset needs
print("\n=== TrainingDataset schema property ===")
from hsfs import training_dataset as td_module
if hasattr(td_module.TrainingDataset, 'schema'):
    td = td_module.TrainingDataset.__new__(td_module.TrainingDataset)
    print(type(td.schema))

# Check init signature
print("\n=== TrainingDataset init signature ===")
print(inspect.signature(td_module.TrainingDataset.__init__))
