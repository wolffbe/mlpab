import inspect
from vertexai.resources.preview import feature_store as fs

print("create_optimized_store:", inspect.signature(fs.FeatureOnlineStore.create_optimized_store))
print("create_bigtable_store:", inspect.signature(fs.FeatureOnlineStore.create_bigtable_store))
print("create_feature_view:", inspect.signature(fs.FeatureOnlineStore.create_feature_view))
print("FG.create_feature:", inspect.signature(fs.FeatureGroup.create_feature))
print("FV.read:", inspect.signature(fs.FeatureView.read))
print("FV.sync:", inspect.signature(fs.FeatureView.sync))
print()
print("create_feature_view doc:")
print(fs.FeatureOnlineStore.create_feature_view.__doc__[:2000])
