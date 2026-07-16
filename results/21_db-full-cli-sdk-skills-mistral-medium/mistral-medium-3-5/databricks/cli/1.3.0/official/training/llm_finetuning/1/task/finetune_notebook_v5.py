# Databricks notebook source
# This is a Databricks notebook that will copy files and run the fine-tuning script

# Cell 1: Download files from workspace using API
import os
import sys
import base64
import requests
import json

os.makedirs('/tmp', exist_ok=True)
sys.path.insert(0, '/tmp')

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
        print(f"Downloaded {filename}, size: {len(content)} bytes")
    else:
        print(f"Failed to download {filename}: {response}")

# Cell 2: Run the fine-tuning script
os.chdir('/tmp')
print(f"Current directory: {os.getcwd()}")

# Run the script
import finetune_model
finetune_model.main()

# Cell 3: Upload output files back to workspace
# Read metrics.json
with open('/tmp/metrics.json', 'r') as f:
    metrics = json.load(f)
    print(f"Metrics: {metrics}")

# Upload metrics.json to workspace
metrics_content = base64.b64encode(open('/tmp/metrics.json', 'rb').read()).decode('utf-8')
url = f"{host}/api/2.0/workspace/import"
params = {
    "path": f"{workspace_path}/metrics.json",
    "format": "RAW",
    "language": "JSON",
    "content": metrics_content,
    "overwrite": "true"
}
response = requests.post(url, headers=headers, json=params).json()
print(f"Uploaded metrics.json: {response}")

# Upload finetuned_model.npz to workspace
with open('/tmp/finetuned_model.npz', 'rb') as f:
    npz_content = f.read()
npz_content_b64 = base64.b64encode(npz_content).decode('utf-8')
params = {
    "path": f"{workspace_path}/finetuned_model.npz",
    "format": "RAW",
    "language": "PYTHON",
    "content": npz_content_b64,
    "overwrite": "true"
}
response = requests.post(url, headers=headers, json=params).json()
print(f"Uploaded finetuned_model.npz: {response}")

print("Done!")
