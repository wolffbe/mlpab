import hopsworks

project = hopsworks.login()
print("project:", project.name)
fs = project.get_feature_store()
for name in ["rawa8af783", "rawb8af783", "derived8af783"]:
    try:
        fg = fs.get_feature_group(name, 1)
        if fg is None:
            print(name, "-> None")
        else:
            print(name, "-> exists, online_enabled:", fg.online_enabled,
                  "features:", [f.name for f in fg.features])
    except Exception as e:
        print(name, "->", type(e).__name__, str(e)[:200])
