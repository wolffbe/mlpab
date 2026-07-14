import os

os.environ["NO_PROXY"] = ""
os.environ["no_proxy"] = ""

import urllib3

urllib3.disable_warnings()

import hopsworks

proj = hopsworks.login(hostname_verification=False)
print("project:", proj.name)
fs = proj.get_feature_store()
print("fs:", fs.name)
