import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

# Read the prediction log
prediction_log = pd.read_csv('/Projects/mlpab07df76/Resources/prediction_log.csv', parse_dates=['ts'])

# Ingest into feature group
fg = fs.get_feature_group('prediction_log', version=1)
fg.insert(prediction_log, write_options={'wait_for_job': True})

# Compute daily statistics
prediction_log['day'] = prediction_log['ts'].dt.date
stats = prediction_log.groupby('day')['prediction'].agg(['mean', 'std']).reset_index()
stats['day'] = pd.to_datetime(stats['day'])

# Save statistics
stats.to_csv('/Projects/mlpab07df76/Resources/prediction_stats.csv', index=False)