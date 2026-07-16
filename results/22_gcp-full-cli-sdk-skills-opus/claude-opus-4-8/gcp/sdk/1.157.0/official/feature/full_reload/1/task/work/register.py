import os
# gRPC cannot use the socks5h proxy in this env; force REST transport.
for v in ("GRPC_PROXY", "grpc_proxy"):
    os.environ.pop(v, None)
import vertexai
from vertexai.resources.preview import feature_store as fs

project = os.environ['GCP_PROJECT']
location = os.environ['GCP_LOCATION']
dataset = os.environ['GCP_BQ_DATASET']
prefix = os.environ['MLPAB_GCP_PREFIX']

vertexai.init(project=project, location=location, api_transport="rest")


def get_or_create(fg_name, table_id, feature_cols):
    uri = f"bq://{project}.{dataset}.{table_id}"
    try:
        fg = fs.FeatureGroup(fg_name)
        # verify it belongs to this run's dataset
        cur = fg.gca_resource.big_query.big_query_source.input_uri
        print(f"exists FeatureGroup {fg_name} -> {cur}")
        assert dataset in cur, f"name collision: {fg_name} points to {cur}"
    except Exception as e:
        if "collision" in str(e):
            raise
        src = fs.FeatureGroupBigQuerySource(uri=uri, entity_id_columns=["row_id"])
        fg = fs.FeatureGroup.create(name=fg_name, source=src)
        print(f"created FeatureGroup {fg_name} -> {uri} key=row_id")
    existing = {f.name for f in fg.list_features()}
    for c in feature_cols:
        if c in existing:
            print(f"  feature {c} (exists)")
        else:
            fg.create_feature(name=c, version_column_name=c)
            print(f"  feature {c} (created)")
    return fg


# Version 1 (initial export)
get_or_create(f"{prefix}_customerscd1186_1", "customerscd1186_1",
              ["name", "balance_eur", "updated_at"])

# Version 2 (graded deliverable)
get_or_create(f"{prefix}_customerscd1186_2", "customerscd1186_2",
              ["full_name", "balance", "currency", "updated_at"])
