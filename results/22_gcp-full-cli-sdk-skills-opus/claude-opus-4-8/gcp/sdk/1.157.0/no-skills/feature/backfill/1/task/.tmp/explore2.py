import importlib
for modpath in [
    "vertexai",
    "vertexai.resources.preview.feature_store",
    "vertexai.resources.preview",
]:
    try:
        mod = importlib.import_module(modpath)
        names = [x for x in dir(mod) if not x.startswith("_")]
        print(modpath, "->", names)
    except Exception as e:
        print(modpath, "ERR", repr(e))
