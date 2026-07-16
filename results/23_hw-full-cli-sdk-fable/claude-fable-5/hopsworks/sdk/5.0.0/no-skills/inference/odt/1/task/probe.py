import hopsworks
print(hopsworks.__version__)
print([n for n in dir(hopsworks) if not n.startswith('_')])
help(hopsworks.login)
