import hopsworks

hopsworks.login()
dataset_api = hopsworks.project.Project().get_dataset_api()

# Upload the answer file
upload_path = dataset_api.upload(
    local_path="submission/answers.json",
    upload_path="submission/answers.json",
    overwrite=True
)
print(f"Answer uploaded to: {upload_path}")
