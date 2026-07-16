import os

for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

fv = fs.get_feature_view("churntraining30fee3", 1)
print("fv features:")
for f in fv.features:
    print(" ", f.name, f.type, "label=" + str(f.label))

td = fv.get_training_datasets()[0]
print("td schema:")
try:
    for f in td.schema:
        print(" ", f.name, f.type)
except Exception as e:
    print("  no schema:", e)
