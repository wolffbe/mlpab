import os
import vertexai
from vertexai.resources.preview import feature_store as vfs
from vertexai.resources.preview.feature_store import utils as fsutils
import google.cloud.aiplatform as aiplatform

proj = os.environ['GCP_PROJECT']; ds = os.environ['GCP_BQ_DATASET']; loc = os.environ['GCP_LOCATION']

vertexai.init(project=proj, location=loc, api_transport="rest")
aiplatform.init(project=proj, location=loc, api_transport="rest")

def bq_uri(table):
    return f"bq://{proj}.{ds}.{table}"

specs = [
    ("rawa55c41b", "a_val"),
    ("rawb55c41b", "b_val"),
    ("derived55c41b", "col_sum"),
]

# Delete any stale feature groups (e.g. from a prior run pointing at another dataset)
for name, _ in specs:
    try:
        fg = vfs.FeatureGroup(name)
        cur = getattr(fg.source, "uri", "")
        if f"{proj}.{ds}." not in (cur or ""):
            print(f"deleting stale {name} (source={cur})")
            fg.delete(force=True)
        else:
            print(f"{name} already points at current dataset ({cur})")
    except Exception as e:
        print("no existing", name, repr(e)[:120])

fgs = {}
for name, col in specs:
    src = fsutils.FeatureGroupBigQuerySource(uri=bq_uri(name), entity_id_columns=["row_id"])
    try:
        fg = vfs.FeatureGroup.create(name=name, source=src, labels={"version": "1"},
                                     description=f"Feature table {name} v1, record key row_id")
        print("created FeatureGroup", name)
    except Exception as e:
        print("FeatureGroup create failed", name, repr(e)[:200])
        fg = vfs.FeatureGroup(name)
    fgs[name] = fg
    existing = [f.name for f in fg.list_features()]
    if col not in existing:
        fg.create_feature(name=col)
        print("  created feature", col)
    else:
        print("  feature exists", col)

for n in fgs:
    fg = fgs[n]
    print(n, "->", getattr(fg.source, "uri", None), [f.name for f in fg.list_features()])
