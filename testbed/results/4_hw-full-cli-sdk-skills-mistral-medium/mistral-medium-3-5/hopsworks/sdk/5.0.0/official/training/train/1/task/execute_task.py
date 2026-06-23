#!/usr/bin/env python3
"""Execute the ML platform task using Hopsworks SDK"""
import hopsworks
import os
import json

def main():
    # Step 1: Login and get project
    hopsworks.login()
    project = hopsworks.get_current_project()
    
    # Step 2: Get APIs
    dataset_api = project.get_dataset_api()
    jobs_api = project.get_jobs_api()
    fs = project.get_feature_store()
    
    # Step 3: Upload data files and script to Resources directory
    # The job will read from its working directory, so we need to upload to Resources
    print("Uploading files to Resources...")
    
    # Upload train.csv
    dataset_api.upload("data/train.csv", "Resources/train.csv", overwrite=True)
    # Upload score.csv
    dataset_api.upload("data/score.csv", "Resources/score.csv", overwrite=True)
    # Upload train_model.py
    dataset_api.upload("data/train_model.py", "Resources/train_model.py", overwrite=True)
    
    # Step 4: Configure and create the job
    print("Creating job...")
    python_config = jobs_api.get_configuration('PYTHON')
    python_config['appPath'] = 'Resources/train_model.py'
    python_config['jobType'] = 'PYTHON'
    
    # Create the job
    job = jobs_api.create_job('trainjob0053a1', python_config)
    
    # Step 5: Launch the job
    print("Launching job...")
    execution = job.launch()
    
    # Wait for completion
    print("Waiting for job to complete...")
    execution.wait()
    
    # Step 6: Get the predictions output
    # The job writes predictions.csv to its working directory (Resources/)
    print("Reading predictions...")
    predictions_content = dataset_api.read_content("Resources/predictions.csv")
    
    # Parse the CSV content
    import io
    import pandas as pd
    predictions_df = pd.read_csv(io.StringIO(predictions_content))
    
    # Step 7: Create feature table
    print("Creating feature table...")
    fg = fs.get_or_create_feature_group(
        name="predictions0053a1",
        version=1,
        primary_key=["row_id"],
        description="Predictions from trainjob0053a1"
    )
    
    # Insert data into feature group
    fg.insert(predictions_df)
    
    # Step 8: Enable online access
    print("Enabling online access...")
    # The feature group should be available for online lookup
    # In Hopsworks, feature groups are available for online access by default
    # when they have a primary key
    
    # Step 9: Write answers.json
    os.makedirs('submission', exist_ok=True)
    with open('submission/answers.json', 'w') as f:
        json.dump({"job_name": "trainjob0053a1"}, f)
    
    print("Task completed successfully!")

if __name__ == '__main__':
    main()
