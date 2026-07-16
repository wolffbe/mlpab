import hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

from hsfs.core import arrow_flight_client
afc = arrow_flight_client.get_instance()
print("FLIGHT methods:", [m for m in dir(afc) if not m.startswith("__")])
print("supported:", getattr(afc, "is_query_supported", None))
