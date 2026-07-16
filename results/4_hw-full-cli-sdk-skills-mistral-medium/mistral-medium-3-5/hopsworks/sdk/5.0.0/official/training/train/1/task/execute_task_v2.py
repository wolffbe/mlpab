#!/usr/bin/env python3
"""Execute the ML platform task using Hopsworks SDK"""
import hopsworks
import os
import json
import csv
import io

def main():
    # Step 1: Login and get project
    hopsworks.login()
    project = hopsworks.get_current_project()
    
    # Step 2: Get APIs
    dataset_api = project.get_dataset_api()
    jobs_api = project.get_jobs_api()
    fs = project.get_feature_store()
    
    # Step 3: Upload data files and script to Resources directory
    print("Uploading files to Resources...")
    dataset_api.upload("data/train.csv", "Resources/train.csv", overwrite=True)
    dataset_api.upload("data/score.csv", "Resources/score.csv", overwrite=True)
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
    print("Reading predictions...")
    predictions_content = dataset_api.read_content("Resources/predictions.csv")
    
    # Parse CSV manually using csv module (standard library, not ML library)
    csv_reader = csv.DictReader(io.StringIO(predictions_content))
    rows = []
    for row in csv_reader:
        rows.append(row)
    
    # Convert to list of lists for insert
    # Get headers
    headers = list(rows[0].keys()) if rows else []
    data = [[row[h] for h in headers] for row in rows]
    
    # Step 7: Create feature table with online access
    print("Creating feature table...")
    fg = fs.get_or_create_feature_group(
        name="predictions0053a1",
        version=1,
        primary_key=["row_id"],
        online_enabled=True,  # Enable online/real-time access
        description="Predictions from trainjob0053a1"
    )
    
    # Insert data into feature group
    # insert expects features as DataFrame or list of lists
    fg.insert(data, wait=True)
    
    # Step 8: Write answers.json
    os.makedirs('submission', exist_ok=True)
    with open('submission/answers.json', 'w') as f:
        json.dump({"job_name": "trainjob0053a1"}, f)
    
    print("Task completed successfully!")

if __name__ == '__main__':
    main()
