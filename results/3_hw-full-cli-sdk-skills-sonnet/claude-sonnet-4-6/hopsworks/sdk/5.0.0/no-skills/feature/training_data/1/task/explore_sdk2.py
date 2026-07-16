import hopsworks
import inspect

print("Connecting...")
project = hopsworks.login()
fs = project.get_feature_store()

print("\n=== create_training_dataset - full help ===")
help(fs.create_training_dataset)

print("\n=== TrainingDataset class ===")
from hsfs import training_dataset
td_methods = [m for m in dir(training_dataset.TrainingDataset) if not m.startswith('_')]
print(td_methods)

if hasattr(training_dataset.TrainingDataset, 'save'):
    print("\nsave signature:", inspect.signature(training_dataset.TrainingDataset.save))
if hasattr(training_dataset.TrainingDataset, 'insert'):
    print("\ninsert signature:", inspect.signature(training_dataset.TrainingDataset.insert))
