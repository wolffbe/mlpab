import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

fg_a = fs.get_feature_group("rawa8af783", 1)
fg_b = fs.get_feature_group("rawb8af783", 1)
print("fg_a:", fg_a is not None, "fg_b:", fg_b is not None)

df = fg_a.read()
print("rawa rows:", len(df))
print(df.head())
