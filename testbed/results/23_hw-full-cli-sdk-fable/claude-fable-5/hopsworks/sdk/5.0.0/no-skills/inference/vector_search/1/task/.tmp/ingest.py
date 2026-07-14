import csv
import json
import os

os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

import hopsworks
import pandas as pd
import hsfs.engine as hsfs_engine
import hsfs.util as hsfs_util
import hopsworks_common.core.dataset_api as dataset_api_mod

# The legacy upload path passes the FeatureGroup object where a filename str is
# expected (SDK bug: TypeError in _get_flow_base_params); coerce it to the name.
_orig_flow_base = dataset_api_mod.DatasetApi._get_flow_base_params


def _fixed_flow_base(self, file_name, num_chunks, size, chunk_size):
    if not isinstance(file_name, str):
        file_name = hsfs_util.feature_group_name(file_name)
    return _orig_flow_base(self, file_name, num_chunks, size, chunk_size)


dataset_api_mod.DatasetApi._get_flow_base_params = _fixed_flow_base

project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_feature_group("itemsf57ff6", version=1)
print("fg:", fg.name, fg.version, "embedding_index:", fg.embedding_index)

items = []
with open("data/items.csv") as f:
    for row in csv.DictReader(f):
        items.append(
            {
                "item_id": row["item_id"],
                "embedding": json.loads(row["embedding"]),
                "label": row["label"],
            }
        )
df = pd.DataFrame(items)
print("items:", len(df))

# Direct HDFS (delta-rs) and Kafka writes are unreachable from this sandbox;
# route the insert through the platform-side ingestion job instead, which only
# needs HTTPS (dataset upload + job REST API).
eng = hsfs_engine.get_instance()


def save_via_ingestion_job(
    feature_group,
    dataframe,
    operation,
    online_enabled,
    storage,
    offline_write_options,
    online_write_options,
    validation_id=None,
):
    return eng.legacy_save_dataframe(
        feature_group,
        dataframe,
        operation,
        online_enabled,
        storage,
        offline_write_options,
        online_write_options,
        validation_id,
    )


eng.save_dataframe = save_via_ingestion_job

job = fg.insert(df, write_options={"wait_for_job": True})
print("ingestion job finished:", job)
