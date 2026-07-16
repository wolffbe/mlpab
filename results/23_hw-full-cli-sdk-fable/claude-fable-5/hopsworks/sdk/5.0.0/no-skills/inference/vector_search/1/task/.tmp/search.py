import csv
import json
import os
import time

os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

import hopsworks
import hopsworks_common.core.opensearch_api as osapi_mod
from opensearchpy import RequestsHttpConnection

# Direct TCP to the OpenSearch endpoint is not reachable from this sandbox;
# use the requests-based connection class so the HTTPS proxy env vars apply,
# and skip cert verification (IP-address certificate).
_orig_cfg = osapi_mod.OpenSearchApi.get_default_py_config


def _proxied_cfg(self, feature_store_id=None):
    cfg = _orig_cfg(self, feature_store_id)
    cfg["connection_class"] = RequestsHttpConnection
    cfg["verify_certs"] = False
    cfg["ssl_show_warn"] = False
    cfg.pop("ca_certs", None)
    cfg.pop("ssl_assert_hostname", None)
    return cfg


osapi_mod.OpenSearchApi.get_default_py_config = _proxied_cfg

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("itemsf57ff6", version=1)
feature_names = [f.name for f in fg.features]
print("features:", feature_names)

queries = []
with open("data/queries.csv") as f:
    for row in csv.DictReader(f):
        queries.append((row["query_id"], json.loads(row["embedding"])))
print("queries:", len(queries))


def item_id_of(row):
    if isinstance(row, dict):
        return row["item_id"]
    return row[feature_names.index("item_id")]


# sanity probe: make sure the index is fully populated before answering
probe = fg.find_neighbors(queries[0][1], k=300)
print("probe hits:", len(probe))
deadline = time.time() + 300
while len(probe) < 300 and time.time() < deadline:
    time.sleep(15)
    probe = fg.find_neighbors(queries[0][1], k=300)
    print("probe hits:", len(probe))

neighbors = {}
for qid, emb in queries:
    res = fg.find_neighbors(emb, k=5)
    assert len(res) == 5, (qid, len(res))
    dists = [round(d, 4) for d, _ in res]
    assert dists == sorted(dists), (qid, dists)
    neighbors[qid] = [item_id_of(row) for _, row in res]
    print(qid, dists, neighbors[qid])

os.makedirs("submission", exist_ok=True)
answer = {"store": "itemsf57ff6", "neighbors": neighbors}
with open("submission/answers.json", "w") as f:
    json.dump(answer, f, indent=1)
print("wrote submission/answers.json")
