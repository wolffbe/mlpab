"""Wrapper script that runs on Hopsworks to execute fine-tuning."""
import os
import subprocess
import hopsworks

project = hopsworks.login()
dataset_api = project.get_dataset_api()

workdir = os.getcwd()
print(f"Working directory: {workdir}")

# Download required files to working directory
for fname in ['base_model.npz', 'finetune.txt', 'eval.txt', 'finetune_model.py']:
    print(f"Downloading {fname}...")
    dataset_api.download(f'Resources/ftjob30461d_data/{fname}', local_path=workdir, overwrite=True)
    print(f"Downloaded {fname}")

# Run the fine-tuning script
print("Running fine-tuning script...")
result = subprocess.run(['python', 'finetune_model.py'], cwd=workdir, check=True,
                        capture_output=True, text=True)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Fine-tuning complete!")

# Upload outputs back to Hopsworks
print("Uploading results...")
dataset_api.upload(os.path.join(workdir, 'finetuned_model.npz'),
                   'Resources/ftjob30461d_data/', overwrite=True)
dataset_api.upload(os.path.join(workdir, 'metrics.json'),
                   'Resources/ftjob30461d_data/', overwrite=True)
print("Done!")
