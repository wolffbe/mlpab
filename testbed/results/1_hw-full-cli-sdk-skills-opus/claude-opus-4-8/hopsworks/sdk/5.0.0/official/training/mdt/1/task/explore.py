import hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
print("PROJECT:", project.name)
print("FS:", fs.name)
import hsfs
print("hsfs version:", hsfs.__version__)
try:
    from hsfs import engine
    print("ENGINE TYPE:", engine.get_type())
except Exception as e:
    print("engine err:", e)
