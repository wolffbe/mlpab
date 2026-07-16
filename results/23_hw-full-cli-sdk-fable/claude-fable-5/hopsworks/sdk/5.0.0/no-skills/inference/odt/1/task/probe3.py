import hopsworks
import importlib, inspect
hu = importlib.import_module("hsfs.hopsworks_udf")
print([n for n in dir(hu) if not n.startswith('_')])
print(inspect.signature(hu.udf))
print(hu.udf.__doc__[:2500])
U = hu.UDFExecutionMode if hasattr(hu, "UDFExecutionMode") else None
print(U)
HW = hu.HopsworksUdf
print([n for n in dir(HW) if not n.startswith('_')])
