import math

import pandas as pd
import hopsworks
from hsfs import engine, util

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("profiles926b2c", version=1)
print("FG:", fg.name, fg.version, "online:", fg.online_enabled)

df = pd.read_csv("data/features.csv")
df["account_id"] = df["account_id"].astype(str)
for c in ["f1", "f2", "f3", "f4"]:
    df[c] = df[c].astype(float)

eng = engine.get_instance()
app_options = eng._get_app_options({})

print("Configuring ingestion job...")
ingestion_job = eng._feature_group_api.ingestion(fg, app_options)
print("Data path:", ingestion_job.data_path)

dataset_api = eng._dataset_api
df_parquet = df.to_parquet(index=False)
parquet_length = len(df_parquet)
chunk_size = dataset_api.DEFAULT_FLOW_CHUNK_SIZE
num_chunks = math.ceil(parquet_length / chunk_size)
fg_filename = util.feature_group_name(fg)

base_params = dataset_api._get_flow_base_params(
    fg_filename, num_chunks, parquet_length, chunk_size
)
chunk_number = 1
for i in range(0, parquet_length, chunk_size):
    query_params = dict(base_params)
    query_params["flowCurrentChunkSize"] = len(df_parquet[i : i + chunk_size])
    query_params["flowChunkNumber"] = chunk_number
    dataset_api._upload_request(
        query_params,
        ingestion_job.data_path,
        fg_filename,
        df_parquet[i : i + chunk_size],
    )
    chunk_number += 1
print("Upload done, running ingestion job...")

ingestion_job.job.run(await_termination=True)
print("Ingestion job finished:", ingestion_job.job.name)
