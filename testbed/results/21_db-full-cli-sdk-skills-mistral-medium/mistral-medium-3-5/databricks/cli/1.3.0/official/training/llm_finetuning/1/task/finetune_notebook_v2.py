# Databricks notebook source
# This is a Databricks notebook that will copy files and run the fine-tuning script

# Cell 1: Download files from workspace using API
import os
import base64
import requests

os.makedirs('/tmp', exist_ok=True)

# Get files from workspace
host = os.environ.get('DATABRICKS_HOST', 'https://dbc-2a4591fe-28e4.cloud.databricks.com')
token = os.environ.get('DATABRICKS_TOKEN')
workspace_path = "/Users/benedict@logicalclocks.com/mlpab64367b"

files = ['finetune_model.py', 'base_model.npz', 'finetune.txt', 'eval.txt']
headers = {"Authorization": f"Bearer {token}"}

for filename in files:
    url = f"{host}/api/2.0/workspace/export"
    params = {"path": f"{workspace_path}/{filename}", "format": "RAW"}
    response = requests.get(url, headers=headers, params=params).json()
    if 'content' in response:
        content = base64.b64decode(response['content'])
        with open(f'/tmp/{filename}', 'wb') as f:
            f.write(content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}: {response}")

# Cell 2: Run the fine-tuning script
os.chdir('/tmp')
import finetune_model
finetune_model.main()
