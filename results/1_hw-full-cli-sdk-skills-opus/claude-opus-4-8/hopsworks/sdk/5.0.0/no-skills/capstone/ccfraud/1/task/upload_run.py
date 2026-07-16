import hopsworks

proj = hopsworks.login()
ja = proj.get_job_api()
ds = proj.get_dataset_api()

up = ds.upload('train_score_job.py', 'Resources', overwrite=True)
print('uploaded to', up)

cfg = ja.get_configuration('PYTHON')
cfg['appPath'] = 'Resources/train_score_job.py'
cfg['resourceConfig']['memory'] = 8192
cfg['resourceConfig']['cores'] = 2.0
print('config', cfg)

job = ja.create_job('ccfraudfe5424job', cfg)
print('created job', job.name)

execution = job.run(await_termination=True)
print('execution state', execution.state, 'final status', execution.final_status,
      'success', getattr(execution, 'success', None))

try:
    out, err = execution.download_logs()
    print('OUT LOG:', out)
    print('ERR LOG:', err)
except Exception as e:
    print('log download failed', e)
