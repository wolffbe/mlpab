import google.cloud.aiplatform as aiplatform
print("featurestore submodule:", [n for n in dir(aiplatform.featurestore) if not n.startswith('_')])
print("Featurestore:", [n for n in dir(aiplatform.Featurestore) if not n.startswith('_')])
print("EntityType:", [n for n in dir(aiplatform.featurestore.EntityType) if not n.startswith('_')])
