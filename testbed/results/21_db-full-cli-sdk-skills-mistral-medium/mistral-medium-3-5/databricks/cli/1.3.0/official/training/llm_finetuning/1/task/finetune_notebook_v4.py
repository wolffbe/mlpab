# Databricks notebook source
# This is a Databricks notebook that will copy files and run the fine-tuning script

# Cell 1: Download files from workspace using API
import os
import sys
import base64
import requests

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

# List files in /tmp
print("Files in /tmp:")
for f in os.listdir('/tmp'):
    print(f"  {f}")

# Cell 2: Run the fine-tuning script
os.chdir('/tmp')
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Try to import and run
try:
    import finetune_model
    print("Successfully imported finetune_model")
    finetune_model.main()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
