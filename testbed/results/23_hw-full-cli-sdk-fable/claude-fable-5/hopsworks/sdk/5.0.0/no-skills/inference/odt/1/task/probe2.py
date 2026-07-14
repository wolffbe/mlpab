import hopsworks
import importlib
hsfs = importlib.import_module("hsfs")
print("hsfs:", hsfs.__version__ if hasattr(hsfs, "__version__") else "?")
print([n for n in dir(hsfs) if not n.startswith('_')])
u = getattr(hsfs, "udf", None)
print("udf:", u)
if u:
    import inspect
    print(inspect.signature(u))
    print(u.__doc__[:3000] if u.__doc__ else "no doc")
