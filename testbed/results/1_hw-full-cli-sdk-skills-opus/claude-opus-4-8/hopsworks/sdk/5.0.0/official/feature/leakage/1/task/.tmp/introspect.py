import hopsworks
project = hopsworks.login()
print("PROJECT NAME:", project.name)
print("=== project methods (trino/hive/sql/presto) ===")
for m in dir(project):
    if any(k in m.lower() for k in ["trino","hive","sql","presto","jdbc","query","conn"]):
        print("  P:", m)
fs = project.get_feature_store()
print("=== fs methods (sql/query/trino) ===")
for m in dir(fs):
    if any(k in m.lower() for k in ["trino","hive","sql","presto","jdbc","query"]):
        print("  FS:", m)
import hopsworks as h
print("hopsworks version:", getattr(h, "__version__", "?"))
