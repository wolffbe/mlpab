import hopsworks
import inspect

project = hopsworks.login()
fs = project.get_feature_store()

print("=== get_feature_group signature ===")
print(inspect.signature(fs.get_feature_group))

# Try getting feature groups
print("\n=== Trying to get labels_churn ===")
result = fs.get_feature_group("labels_churn", version=1)
print(f"Result: {result}")
print(f"Type: {type(result)}")

if result is not None:
    print(f"Name: {result.name}")
    print(f"Columns: {result.column_names}")

# Try get_feature_groups
print("\n=== get_feature_groups signature ===")
print(inspect.signature(fs.get_feature_groups))
fgs = fs.get_feature_groups("labels_churn")
print(f"Feature groups: {fgs}")
