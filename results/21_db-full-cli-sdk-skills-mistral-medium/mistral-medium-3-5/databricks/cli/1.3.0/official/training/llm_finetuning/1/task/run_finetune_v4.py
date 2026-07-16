#!/usr/bin/env python
# Databricks notebook script
# Use dbutils to access workspace files

# First, let's try to use the workspace API to get the file content
import requests
import os
import json

# Get the workspace host and token from environment
import os
host = os.environ.get('DATABRICKS_HOST', 'https://dbc-2a4591fe-28e4.cloud.databricks.com')
token = os.environ.get('DATABRICKS_TOKEN')

# Get the current user's workspace path
workspace_path = "/Users/benedict@logicalclocks.com/mlpab64367b"

# Download the files to /dbfs/tmp/
import urllib.request

files_to_download = ['finetune_model.py', 'base_model.npz', 'finetune.txt', 'eval.txt']
for filename in files_to_download:
    url = f"{host}/api/2.0/workspace/export"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"path": f"{workspace_path}/{filename}", "format": "RAW"}
    
    try:
        # For .npz files, we need to use a different approach
        if filename.endswith('.npz'):
            # Use dbutils.fs.cp or workspace API to get the file
            # Try using the workspace export API
            import base64
            response = requests.get(url, headers=headers, params=data).json()
            if 'content' in response:
                content = base64.b64decode(response['content'])
                with open(f'/dbfs/tmp/{filename}', 'wb') as f:
                    f.write(content)
        else:
            response = requests.get(url, headers=headers, params=data).json()
            if 'content' in response:
                content = base64.b64decode(response['content'])
                with open(f'/dbfs/tmp/{filename}', 'wb') as f:
                    f.write(content)
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

# Change to the tmp directory
os.chdir('/dbfs/tmp/')

# Now run the script
import finetune_model
if __name__ == "__main__":
    finetune_model.main()
