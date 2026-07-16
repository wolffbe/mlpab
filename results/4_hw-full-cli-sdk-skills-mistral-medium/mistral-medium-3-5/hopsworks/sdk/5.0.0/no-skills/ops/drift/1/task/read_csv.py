import hopsworks

hopsworks.login()
dataset_api = hopsworks.project.Project().get_dataset_api()

# Upload the CSV first
upload_path = dataset_api.upload(
    local_path="data/features.csv",
    upload_path="Resources/drift_data/features.csv",
    overwrite=True
)
print(f"CSV uploaded to: {upload_path}")

# Try to read it back
try:
    content = dataset_api.read_content(upload_path)
    print(f"Content length: {len(content)}")
    print(f"First 500 chars: {content[:500]}")
except Exception as e:
    print(f"Error reading: {e}")
