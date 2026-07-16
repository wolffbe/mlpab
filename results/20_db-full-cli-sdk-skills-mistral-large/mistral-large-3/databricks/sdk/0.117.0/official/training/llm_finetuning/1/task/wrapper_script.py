"""
Wrapper script to run finetune_model.py with DBFS support.
Reads inputs from DBFS and writes outputs to DBFS.
"""
import os
import shutil
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--working_dir', type=str, required=True, help='DBFS directory to read/write files')
args = parser.parse_args()

# Copy inputs from DBFS to local working directory
os.makedirs('/tmp/finetune', exist_ok=True)
for filename in ['base_model.npz', 'finetune.txt', 'eval.txt']:
    dbfs_path = os.path.join(args.working_dir, filename)
    local_path = os.path.join('/tmp/finetune', filename)
    os.system(f'dbfs cp {dbfs_path} {local_path}')

# Run the fine-tuning script
os.chdir('/tmp/finetune')
import finetune_model
finetune_model.main()

# Copy outputs back to DBFS
for filename in ['finetuned_model.npz', 'metrics.json']:
    local_path = os.path.join('/tmp/finetune', filename)
    dbfs_path = os.path.join(args.working_dir, filename)
    os.system(f'dbfs cp {local_path} {dbfs_path}')