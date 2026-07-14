import inspect
from vertexai.resources.preview import feature_store as fs

for name in ["FeatureGroup", "FeatureOnlineStore", "FeatureView"]:
    cls = getattr(fs, name)
    print("="*60)
    print(name, "create sig:")
    try:
        print(inspect.signature(cls.create))
    except Exception as e:
        print("  ", e)

print("="*60)
print("FeatureGroupBigQuerySource:", inspect.signature(fs.FeatureGroupBigQuerySource.__init__))
print("FeatureViewBigQuerySource:", inspect.signature(fs.FeatureViewBigQuerySource.__init__))
print("="*60)
# FeatureView creation is usually via FeatureOnlineStore.create_feature_view
print("FOS methods:", [m for m in dir(fs.FeatureOnlineStore) if not m.startswith('_')])
print("FG methods:", [m for m in dir(fs.FeatureGroup) if not m.startswith('_')])
print("FV methods:", [m for m in dir(fs.FeatureView) if not m.startswith('_')])
