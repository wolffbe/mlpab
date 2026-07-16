import hopsworks
import json
import os

# Login to Hopsworks
hopsworks.login()

# Upload the CSV file
dataset_api = hopsworks.project.Project().get_dataset_api()
upload_path = dataset_api.upload(
    local_path="data/features.csv",
    upload_path="Resources/drift_data/features.csv",
    overwrite=True
)
print(f"CSV uploaded to: {upload_path}")

# Now use Trino to query the data
trino_api = hopsworks.project.Project().get_trino_api()

# Get Trino connection details
try:
    engine = trino_api.create_engine(verify=False)
    print(f"Trino engine created: {engine}")
    
    # Query the data to understand its structure
    query = """
    SELECT * 
    FROM hopsfs."Resources/drift_data/features.csv/features.csv"
    LIMIT 10
    """
    result = engine.execute(query)
    print(f"Query result: {result.fetchall()}")
except Exception as e:
    print(f"Error with Trino: {e}")
    import traceback
    traceback.print_exc()

print("Done!")
