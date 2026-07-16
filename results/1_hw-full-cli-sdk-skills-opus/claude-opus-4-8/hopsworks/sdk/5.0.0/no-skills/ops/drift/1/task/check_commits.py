import warnings; warnings.filterwarnings('ignore')
import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()
fg = fs.get_feature_group('drift_features', version=1)

cd = fg.commit_details()
print('num commits', len(cd))
# print a few
items = sorted(cd.items())
for k, v in items[:3]:
    print(k, v)
print('...')
for k, v in items[-3:]:
    print(k, v)

allstats = fg.get_all_statistics()
print('get_all_statistics type', type(allstats))
