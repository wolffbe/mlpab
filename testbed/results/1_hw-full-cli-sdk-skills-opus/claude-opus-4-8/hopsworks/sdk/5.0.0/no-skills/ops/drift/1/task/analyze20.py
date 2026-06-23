import warnings; warnings.filterwarnings('ignore')
import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()
fg = fs.get_feature_group('drift_features', version=1)
feats = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']

cd = fg.commit_details()
cd = {int(k): v for k, v in cd.items()}
commit_times = sorted(cd.keys())

allstats = fg.get_all_statistics()
# index detail by window_end_commit_time
by_end = {}
for s in allstats:
    by_end.setdefault(s.window_end_commit_time, s.computation_time)

print('day commitDate   ' + '  '.join(f'{f}m/{f}sd' for f in feats))
for i, ct in enumerate(commit_times):
    comp = by_end.get(ct)
    if comp is None:
        print(i, 'no-stats-for-commit', ct)
        continue
    full = fg.get_statistics(computation_time=comp)
    d = {x.feature_name: (x.mean, x.stddev) for x in full.feature_descriptive_statistics}
    line = '  '.join(f'{d[f][0]:.2f}/{d[f][1]:.2f}' for f in feats)
    print(f'{i+1:2d} {cd[ct]["committedOn"][:14]} {line}')
