import hopsworks

hopsworks.login()
dataset_api = hopsworks.project.Project().get_dataset_api()

# Create the submission directory first
try:
    dataset_api.mkdir("submission")
    print("Created submission directory")
except Exception as e:
    print(f"Error creating directory: {e}")

# Upload the answer file to submission
upload_path = dataset_api.upload(
    local_path="submission/answers.json",
    upload_path="submission/answers.json",
    overwrite=True
)
print(f"Answer uploaded to: {upload_path}")
