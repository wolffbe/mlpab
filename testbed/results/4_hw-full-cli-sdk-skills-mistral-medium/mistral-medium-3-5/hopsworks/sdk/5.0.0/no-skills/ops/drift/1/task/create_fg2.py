import hopsworks
import hsfs

hopsworks.login()
proj = hopsworks.project.Project()
fs = proj.get_feature_store()
dataset_api = proj.get_dataset_api()

# First create the feature group
fg = fs.create_feature_group(
    name="drift_detection_fg",
    version=1,
    description="Feature group for drift detection",
    online_enabled=False,
    time_travel_format="HUDI",
    event_time="event_time"
)
print(f"Created feature group: {fg.name} v{fg.version}")
print(f"Feature group ID: {fg.id}")

# Now use the dataset API to upload data to the feature group
try:
    result = dataset_api.upload_feature_group(
        feature_group=fg,
        path="data/features.csv",
        dataframe=None
    )
    print(f"Upload result: {result}")
except Exception as e:
    print(f"Error uploading feature group: {e}")
    import traceback
    traceback.print_exc()

# List feature groups
fgs = fs.get_feature_groups()
print(f"Number of feature groups: {len(fgs)}")
for fg_item in fgs:
    print(f"  - {fg_item.name} v{fg_item.version} ID={fg_item.id}")
