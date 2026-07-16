import inspect
from vertexai.resources.preview import feature_store as fs
print("FG.delete:", inspect.signature(fs.FeatureGroup.delete))
print("FG has update?:", [m for m in dir(fs.FeatureGroup) if 'update' in m.lower()])
