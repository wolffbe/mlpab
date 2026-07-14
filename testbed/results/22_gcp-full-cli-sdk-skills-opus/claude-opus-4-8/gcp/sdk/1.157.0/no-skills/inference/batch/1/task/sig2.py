import inspect
import google.cloud.aiplatform as aiplatform
print("init params:", list(inspect.signature(aiplatform.init).parameters))
# Featurestore client_class / how transport is chosen
import google.cloud.aiplatform.initializer as initz
print("has api_transport attr on init:", "api_transport" in inspect.signature(aiplatform.init).parameters)
