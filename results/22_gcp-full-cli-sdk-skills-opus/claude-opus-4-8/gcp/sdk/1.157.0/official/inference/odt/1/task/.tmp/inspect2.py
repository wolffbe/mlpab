import inspect
from vertexai.resources.preview import feature_store as fs

print("create_optimized_store:", inspect.signature(fs.FeatureOnlineStore.create_optimized_store))
print("create_bigtable_store:", inspect.signature(fs.FeatureOnlineStore.create_bigtable_store))
print("FeatureView.sync:", [m for m in dir(fs.FeatureView) if not m.startswith("_")])
