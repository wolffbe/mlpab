import inspect
import google.cloud.aiplatform as aiplatform
print("Featurestore.create:", inspect.signature(aiplatform.Featurestore.create))
print("create_entity_type:", inspect.signature(aiplatform.Featurestore.create_entity_type))
print("EntityType.create_feature:", inspect.signature(aiplatform.featurestore.EntityType.create_feature))
print("ingest_from_bq:", inspect.signature(aiplatform.featurestore.EntityType.ingest_from_bq))
print("EntityType.read:", inspect.signature(aiplatform.featurestore.EntityType.read))
print("update_online_store:", inspect.signature(aiplatform.Featurestore.update_online_store))
