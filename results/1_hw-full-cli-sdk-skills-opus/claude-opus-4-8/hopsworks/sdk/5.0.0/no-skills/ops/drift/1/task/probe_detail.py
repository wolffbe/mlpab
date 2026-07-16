import warnings; warnings.filterwarnings('ignore')
import inspect
import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()
fg = fs.get_feature_group('drift_features', version=1)

print('get_statistics sig', inspect.signature(fg.get_statistics))
allstats = fg.get_all_statistics()
allstats.sort(key=lambda s: (s.computation_time or 0))
feats = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']

for idx in [0, len(allstats) // 2, len(allstats) - 1]:
    s = allstats[idx]
    full = fg.get_statistics(computation_time=s.computation_time)
    fds = full.feature_descriptive_statistics
    c = {x.feature_name: x.count for x in fds}
    print('idx', idx, 'compTime', s.computation_time, 'f1_count', c.get('f1'))
