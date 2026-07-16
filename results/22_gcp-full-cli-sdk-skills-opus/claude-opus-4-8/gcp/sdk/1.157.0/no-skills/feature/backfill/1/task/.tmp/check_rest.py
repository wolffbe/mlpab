import inspect
import google.cloud.aiplatform as aiplatform
print("init sig:")
print(inspect.signature(aiplatform.init))
from google.cloud.aiplatform import initializer
print("global_config attrs:", [a for a in dir(initializer.global_config) if 'transport' in a.lower()])

# Does the feature store gapic client support rest transport?
from google.cloud.aiplatform_v1.services.feature_registry_service import FeatureRegistryServiceClient
print("FeatureRegistryServiceClient transports:")
print([k for k in FeatureRegistryServiceClient._transport_registry.keys()])
