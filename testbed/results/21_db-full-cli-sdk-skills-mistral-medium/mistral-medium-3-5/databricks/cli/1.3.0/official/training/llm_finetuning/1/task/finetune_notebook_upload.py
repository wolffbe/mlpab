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

# Get files from workspace - try without explicit token first
workspace_path = "/Users/benedict@logicalclocks.com/mlpab64367b"

files = ['finetune_model.py', 'base_model.npz', 'finetune.txt', 'eval.txt']

for filename in files:
    # Try using dbutils to read the file
    try:
        # Use dbutils to read workspace file
        # In Databricks, we can use dbutils.fs.head or dbutils.fs.cat
        # But those are for DBFS, not workspace
        # Let's try using the workspace API with implicit auth
        import os
        host = os.environ.get('DATABRICKS_HOST', 'https://dbc-2a4591fe-28e4.cloud.databricks.com')
        token = os.environ.get('DATABRICKS_TOKEN', '')
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        
        url = f"{host}/api/2.0/workspace/export"
        params = {"path": f"{workspace_path}/{filename}", "format": "RAW"}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'content' in data:
                content = base64.b64decode(data['content'])
                with open(f'/tmp/{filename}', 'wb') as f:
                    f.write(content)
                print(f"Downloaded {filename}")
            else:
                print(f"Failed to download {filename}: {data}")
        else:
            print(f"HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

# Cell 2: Run the fine-tuning script
os.chdir('/tmp')

# Execute the script directly
with open('/tmp/finetune_model.py', 'r') as f:
    script_code = f.read()
exec(script_code)
main()

# Cell 3: Upload output files back to workspace
# Read metrics.json
with open('/tmp/metrics.json', 'r') as f:
    metrics = json.load(f)
    eval_loss = metrics['eval_loss']
    base_eval_loss = metrics['base_eval_loss']
    print(f"eval_loss: {eval_loss}, base_eval_loss: {base_eval_loss}")

# Upload metrics.json to workspace
metrics_content = base64.b64encode(open('/tmp/metrics.json', 'rb').read()).decode('utf-8')
url = f"{host}/api/2.0/workspace/import"
import_params = {
    "path": f"{workspace_path}/metrics.json",
    "format": "RAW",
    "language": "JSON",
    "content": metrics_content,
    "overwrite": "true"
}
response = requests.post(url, headers=headers, json=import_params)
print(f"Uploaded metrics.json: {response.status_code}, {response.text}")

# Upload finetuned_model.npz to workspace
with open('/tmp/finetuned_model.npz', 'rb') as f:
    npz_content = f.read()
npz_content_b64 = base64.b64encode(npz_content).decode('utf-8')
import_params = {
    "path": f"{workspace_path}/finetuned_model.npz",
    "format": "RAW",
    "language": "PYTHON",
    "content": npz_content_b64,
    "overwrite": "true"
}
response = requests.post(url, headers=headers, json=import_params)
print(f"Uploaded finetuned_model.npz: {response.status_code}, {response.text}")

print("Done!")
