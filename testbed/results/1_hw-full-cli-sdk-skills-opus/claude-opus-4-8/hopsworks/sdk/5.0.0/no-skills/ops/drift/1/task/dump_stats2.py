import warnings; warnings.filterwarnings('ignore')
import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()
fg = fs.get_feature_group('drift_features', version=1)

feats = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
allstats = fg.get_all_statistics()
print('count', len(allstats) if allstats else 0)
rows = []
for s in allstats:
    fds = s.feature_descriptive_statistics
    if not fds:
        rows.append((s.computation_time, s.window_start_commit_time, s.window_end_commit_time, None))
        continue
    d = {c.feature_name: (c.mean, c.stddev, c.count) for c in fds}
    rows.append((s.computation_time, s.window_start_commit_time, s.window_end_commit_time, d))

rows.sort(key=lambda r: (r[0] or 0))
print('compTime  winStart  winEnd  ' + '  '.join(f'{f}(mean/cnt)' for f in feats))
for ct, ws, we, d in rows:
    if d is None:
        print(ct, ws, we, 'NO-DETAIL')
        continue
    line = '  '.join(f'{d.get(f, (float("nan"),0,0))[0]:.3f}/{d.get(f,(0,0,0))[2]}' for f in feats)
    print(ct, ws, we, line)
