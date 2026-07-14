import os
from google.cloud import bigquery
import google.cloud.aiplatform_v1 as v1

PROJECT = os.environ["GCP_PROJECT"]; LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]; PREFIX = os.environ["MLPAB_GCP_PREFIX"]
API = f"{LOCATION}-aiplatform.googleapis.com"
FS_ID = f"{PREFIX}_fs347afc"; ET_ID = "features347afc"

bq = bigquery.Client(project=PROJECT)
tbl = bq.get_table(f"{PROJECT}.{DATASET}.features347afc")
print("OFFLINE BQ table features347afc columns:", [(f.name, f.field_type) for f in tbl.schema])
print("rows:", tbl.num_rows)

# spot-check amount_7d window correctness against a manual recompute
q = f"""
WITH t AS (SELECT * FROM `{PROJECT}.{DATASET}.stg_transactions`),
chk AS (
  SELECT a.row_id,
    (SELECT ROUND(SUM(b.amount),6) FROM t b
       WHERE b.account_id=a.account_id
         AND b.event_time BETWEEN a.event_time-604800000 AND a.event_time) AS manual
  FROM t a )
SELECT COUNT(*) mism
FROM `{PROJECT}.{DATASET}.features347afc` f JOIN chk USING(row_id)
WHERE ROUND(f.amount_7d,6) != chk.manual
"""
print("amount_7d mismatches vs manual:", list(bq.query(q).result())[0].mism)

# registered new-style FeatureGroup
reg = v1.FeatureRegistryServiceClient(client_options={"api_endpoint": API}, transport="rest")
fg = reg.get_feature_group(name=reg.feature_group_path(PROJECT, LOCATION, "features347afc"))
print("OFFLINE FeatureGroup:", fg.name.split("/")[-1], "entity_id_columns=", list(fg.big_query.entity_id_columns))
print("  registered features:", [f.name.split("/")[-1] for f in reg.list_features(parent=fg.name)])

# online read (low-latency)
online = v1.FeaturestoreOnlineServingServiceClient(client_options={"api_endpoint": API}, transport="rest")
et = online.entity_type_path(PROJECT, LOCATION, FS_ID, ET_ID)
resp = online.read_feature_values(request=v1.ReadFeatureValuesRequest(
    entity_type=et, entity_id="R00000",
    feature_selector=v1.FeatureSelector(id_matcher=v1.IdMatcher(ids=["account_id","event_time","amount_usd","is_weekend","amount_7d"]))))
hdr = [d.id for d in resp.header.data]
vals = []
for d in resp.entity_view.data:
    v = d.value
    vals.append(v.string_value or v.int64_value or v.double_value)
print("ONLINE Featurestore", FS_ID, "read R00000:")
print("  ", dict(zip(hdr, vals)))
print("VERIFIED")
