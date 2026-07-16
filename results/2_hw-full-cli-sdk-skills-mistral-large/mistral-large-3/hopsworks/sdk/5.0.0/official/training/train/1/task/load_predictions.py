"""
Script to load predictions into a feature table and write the submission file.
"""
import hopsworks
import pandas as pd
import json

# Authenticate with Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Define job name
job_name = "trainjobd2a49d"

# Read predictions into a DataFrame
predictions_path = "predictions.csv"
predictions_df = pd.read_csv(predictions_path)
print(f"Predictions DataFrame shape: {predictions_df.shape}")

# Create or update feature table
feature_table_name = "predictionsd2a49d"
feature_table_version = 1

# Create feature group
fg = fs.create_feature_group(
    name=feature_table_name,
    version=feature_table_version,
    description=f"Predictions from job {job_name}",
    primary_key=["row_id"],
    online_enabled=True  # Enable online access
)

# Insert predictions into feature group
fg.insert(predictions_df)
print(f"Feature table {feature_table_name} version {feature_table_version} created and populated.")

# Write submission file
submission_data = {"job_name": job_name}
submission_path = "submission/answers.json"
with open(submission_path, "w") as f:
    json.dump(submission_data, f)

print(f"Submission file written to {submission_path}")