import inspect
import google.cloud.aiplatform as aiplatform
print("init params:", list(inspect.signature(aiplatform.init).parameters))

# Check REST transport support on the registry/admin clients
from google.cloud.aiplatform_v1.services.feature_registry_service import FeatureRegistryServiceClient
print("registry transports:", FeatureRegistryServiceClient._transport_registry.keys())
from google.cloud.aiplatform_v1.services.feature_online_store_admin_service import FeatureOnlineStoreAdminServiceClient
print("admin transports:", FeatureOnlineStoreAdminServiceClient._transport_registry.keys())
