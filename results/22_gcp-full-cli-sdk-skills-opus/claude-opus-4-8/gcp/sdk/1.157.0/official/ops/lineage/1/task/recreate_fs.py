import os, time
import vertexai
from vertexai.resources.preview import feature_store as vfs
from vertexai.resources.preview.feature_store import utils as fsutils

proj = os.environ['GCP_PROJECT']; ds = os.environ['GCP_BQ_DATASET']; loc = os.environ['GCP_LOCATION']
vertexai.init(project=proj, location=loc, api_transport="rest")

def bq_uri(t): return f"bq://{proj}.{ds}.{t}"
specs = [("rawa55c41b","a_val"),("rawb55c41b","b_val"),("derived55c41b","col_sum")]

fgs = {}
for name, col in specs:
    src = fsutils.FeatureGroupBigQuerySource(uri=bq_uri(name), entity_id_columns=["row_id"])
    fg = None
    for attempt in range(20):
        try:
            existing = vfs.FeatureGroup(name)
            if f"{proj}.{ds}." in (getattr(existing.source,'uri','') or ''):
                fg = existing; print("exists correctly", name); break
        except Exception:
            pass
        try:
            fg = vfs.FeatureGroup.create(name=name, source=src, labels={"version": "1"},
                                         description=f"Feature table {name} v1, record key row_id")
            print("created", name); break
        except Exception as e:
            print(f"  retry {name}: {repr(e)[:120]}"); time.sleep(20)
    fgs[name] = fg
    existing = [f.name for f in fg.list_features()]
    if col not in existing:
        fg.create_feature(name=col, version_column_name=col); print("  created feature", col)
    else:
        print("  feature exists", col)

for n, fg in fgs.items():
    print(n, "->", getattr(fg.source,'uri',None), "features:", [f.name for f in fg.list_features()])
