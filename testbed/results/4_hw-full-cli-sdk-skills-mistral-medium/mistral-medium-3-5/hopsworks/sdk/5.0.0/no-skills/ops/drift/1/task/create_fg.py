import hopsworks

hopsworks.login()
proj = hopsworks.project.Project()
fs = proj.get_feature_store()

# Upload CSV first
dataset_api = proj.get_dataset_api()
upload_path = dataset_api.upload(
    local_path="data/features.csv",
    upload_path="Resources/drift_data/features.csv",
    overwrite=True
)
print(f"CSV uploaded to: {upload_path}")

# Create feature group
fg = fs.create_feature_group(
    name="drift_detection_fg",
    version=1,
    description="Feature group for drift detection",
    online_enabled=False,
    path=upload_path,
    time_travel_format="HUDI"
)
print(f"Created feature group: {fg.name} v{fg.version}")
print(f"Feature group ID: {fg.id}")

# List feature groups
fgs = fs.get_feature_groups()
print(f"Number of feature groups: {len(fgs)}")
