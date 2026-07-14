import hw_env  # noqa: F401
import hopsworks

proj = hopsworks.login()
ds = proj.get_dataset_api()

path = ds.upload(
    "data/features.csv",
    "Resources/drift_task",
    overwrite=True,
    chunk_size=262144,
    simultaneous_chunks=1,
    max_chunk_retries=5,
)
print("upload returned:", path)
print("exists:", ds.exists("Resources/drift_task/features.csv"))
info = ds.get("Resources/drift_task/features.csv")
print("size:", info.get("attributes", {}).get("size"))
