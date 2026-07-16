import os
import google.cloud.aiplatform as aiplatform
import google.cloud.bigquery as bq

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']; ds = os.environ['GCP_BQ_DATASET']
aiplatform.init(project=proj, location=loc, api_transport="rest")
bqc = bq.Client(project=proj)
final = f"{proj}.{ds}.accountsed4daa"

# --- OFFLINE: correctness of latest-revision dedup ---
n, d = list(bqc.query(
    f"SELECT COUNT(*) n, COUNT(DISTINCT row_id) d FROM `{final}`").result())[0]
print(f"OFFLINE rows={n} distinct_row_ids={d} (expect equal, one per row_id)")

# find a row_id that had multiple revisions across batches, confirm latest won
stg = f"{proj}.{ds}.accountsed4daa_staging"
dup = list(bqc.query(f"""
SELECT row_id, COUNT(*) c, MAX(updated_at) latest
FROM `{stg}` GROUP BY row_id HAVING c > 1 ORDER BY c DESC LIMIT 1
""").result())[0]
rid = dup.row_id
print(f"multi-revision row_id={rid} revisions={dup.c} max_updated_at={dup.latest}")
allrev = list(bqc.query(
    f"SELECT status,balance,updated_at FROM `{stg}` WHERE row_id='{rid}' ORDER BY updated_at DESC").result())
print("  all staged revisions:", [(r.status, r.balance, r.updated_at) for r in allrev])
fin = list(bqc.query(
    f"SELECT status,balance,updated_at FROM `{final}` WHERE row_id='{rid}'").result())[0]
print(f"  FINAL table value: status={fin.status} balance={fin.balance} updated_at={fin.updated_at}")
assert fin.updated_at == dup.latest, "final row is not the latest revision!"
print("  -> latest revision correctly retained")

# --- ONLINE: low-latency lookup via legacy Featurestore online serving ---
fstore = aiplatform.Featurestore("accountsed4daa_fs")
et = fstore.get_entity_type("accountsed4daa")
df = et.read(entity_ids=[rid, "R00012"], feature_ids=["status", "balance"])
print("ONLINE read result:")
print(df.to_string(index=False))
print("VERIFY_DONE")
