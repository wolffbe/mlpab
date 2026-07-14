import os
import csv
import google.cloud.aiplatform as aiplatform
from google.cloud import bigquery
from vertexai.resources.preview import feature_store as fs

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]
aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")
bq = bigquery.Client(project=PROJECT)

print("=== BigQuery read-back of graded v2 table (customerscd1186_2) ===")
tbl = bq.get_table(f"{PROJECT}.{DATASET}.customerscd1186_2")
cols = [f.name for f in tbl.schema]
print("columns:", cols)
print("num_rows:", tbl.num_rows)

# compare against source CSV exactly
with open("data/reload/new_export.csv") as fh:
    rdr = csv.DictReader(fh)
    csv_rows = list(rdr)
    csv_cols = rdr.fieldnames
print("csv columns:", csv_cols)
print("csv rows:", len(csv_rows))
print("columns match export exactly:", cols == csv_cols)

# read all rows back from BigQuery and compare keyed by row_id
q = f"SELECT * FROM `{PROJECT}.{DATASET}.customerscd1186_2`"
bq_rows = {r["row_id"]: dict(r) for r in bq.query(q).result()}
csv_map = {r["row_id"]: r for r in csv_rows}
print("row_id sets equal:", set(bq_rows) == set(csv_map))

mismatch = 0
for rid, cr in csv_map.items():
    br = bq_rows.get(rid)
    if br is None:
        mismatch += 1
        continue
    if (br["full_name"] != cr["full_name"] or br["currency"] != cr["currency"]
            or abs(float(br["balance"]) - float(cr["balance"])) > 1e-6
            or int(br["updated_at"]) != int(cr["updated_at"])):
        mismatch += 1
print("value mismatches:", mismatch)

old_cols = {"name", "balance_eur"}
print("no old column names present:", old_cols.isdisjoint(set(cols)))

print()
print("=== Registered FeatureGroups (platform feature tables) ===")
for fg_id in ["customerscd1186_1", "customerscd1186_2"]:
    fg = fs.FeatureGroup(fg_id)
    feats = [f.name for f in fg.list_features()]
    print(f"{fg_id}: source={fg._gca_resource.big_query.big_query_source.input_uri}")
    print(f"   entity_id_columns={list(fg._gca_resource.big_query.entity_id_columns)} features={feats}")
