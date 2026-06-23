import warnings; warnings.filterwarnings('ignore')
import inspect
import hsfs.core.feature_monitoring_config as fmcmod
from hsfs.core.feature_monitoring_config import FeatureMonitoringConfig

print('=== FMC methods ===')
print([m for m in dir(FeatureMonitoringConfig) if not m.startswith('_')])
for m in ['with_detection_window', 'with_reference_window', 'with_reference_value',
          'compare_on', 'save', 'run_feature_monitoring', 'with_detection_window_on',
          'get_history']:
    f = getattr(FeatureMonitoringConfig, m, None)
    if f is not None:
        try:
            print('---', m, inspect.signature(f))
        except Exception as e:
            print('---', m, 'ERR', e)
