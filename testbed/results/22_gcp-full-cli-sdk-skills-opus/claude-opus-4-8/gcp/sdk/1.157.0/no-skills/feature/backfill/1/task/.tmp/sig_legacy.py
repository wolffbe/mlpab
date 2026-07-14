import inspect
import google.cloud.aiplatform as aiplatform
from google.cloud.aiplatform import Featurestore, EntityType, Feature
print("Featurestore.create:", inspect.signature(Featurestore.create))
print("create_entity_type:", inspect.signature(Featurestore.create_entity_type))
print("EntityType.create_feature:", inspect.signature(EntityType.create_feature))
print("ingest_from_bq:", inspect.signature(EntityType.ingest_from_bq))
print("EntityType.read:", inspect.signature(EntityType.read))
