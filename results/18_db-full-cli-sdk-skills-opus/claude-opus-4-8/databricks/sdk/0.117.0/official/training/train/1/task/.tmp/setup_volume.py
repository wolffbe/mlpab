import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType

w = WorkspaceClient()
cat, sch = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
vol = 'trainvola834e5'
try:
    v = w.volumes.create(catalog_name=cat, schema_name=sch, name=vol,
                         volume_type=VolumeType.MANAGED)
    print('created volume', v.full_name)
except Exception as e:
    print('volume create:', repr(e)[:200])

vol_path = f'/Volumes/{cat}/{sch}/{vol}'
for f in ['train.csv', 'score.csv', 'train_model.py']:
    with open(f'data/{f}', 'rb') as fh:
        w.files.upload(f'{vol_path}/{f}', fh, overwrite=True)
    print('uploaded', f)

for d in w.files.list_directory_contents(vol_path):
    print('  ', d.path, d.file_size)
