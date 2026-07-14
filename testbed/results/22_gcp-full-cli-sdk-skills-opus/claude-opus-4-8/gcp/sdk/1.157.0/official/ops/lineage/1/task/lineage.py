import os, json
import vertexai
import google.cloud.aiplatform as aiplatform
from google.cloud import bigquery

proj=os.environ['GCP_PROJECT']; ds=os.environ['GCP_BQ_DATASET']; loc=os.environ['GCP_LOCATION']
prefix=os.environ['MLPAB_GCP_PREFIX']
vertexai.init(project=proj, location=loc, api_transport="rest")
aiplatform.init(project=proj, location=loc, api_transport="rest")
CREDS = aiplatform.initializer.global_config.credentials

os.makedirs("submission", exist_ok=True)

# ---- answers.json ----
sources = sorted(["rawa55c41b", "rawb55c41b"])
with open("submission/answers.json", "w") as f:
    json.dump({"derived_from": sources}, f)
print("wrote answers.json:", sources)

# ---- csv fallback of derived table (read back from platform / BQ) ----
c = bigquery.Client(project=proj)
rows = list(c.query(f"SELECT row_id, col_sum FROM `{proj}.{ds}.derived55c41b` ORDER BY row_id", location=loc).result())
with open("submission/derived55c41b.csv", "w") as f:
    f.write("row_id,col_sum\n")
    for r in rows:
        f.write(f"{r['row_id']},{r['col_sum']}\n")
print("wrote derived55c41b.csv rows:", len(rows))

# ---- lineage via Vertex ML Metadata: sources -> execution -> derived ----
rid_prefix = prefix.replace("_", "-")
def art(rid, table):
    try:
        return aiplatform.Artifact.create(
            schema_title="system.Dataset",
            resource_id=f"{rid_prefix}-{rid}",
            uri=f"bq://{proj}.{ds}.{table}",
            display_name=f"{prefix}_{table}",
            metadata={"featureGroup": table, "dataset": ds},
        )
    except Exception as e:
        print("artifact exists/err", table, repr(e)[:120])
        return aiplatform.Artifact(f"{rid_prefix}-{rid}")

a_src = art("src-rawa", "rawa55c41b")
b_src = art("src-rawb", "rawb55c41b")
d_out = art("out-derived", "derived55c41b")

try:
    exe = aiplatform.Execution.create(
        schema_title="system.Run",
        resource_id=f"{rid_prefix}-derive-col-sum",
        display_name=f"{prefix}_derive_col_sum",
        metadata={"transform": "col_sum = round(a_val + b_val, 6); inner join on row_id"},
        credentials=CREDS,
    )
    exe.assign_input_artifacts([a_src, b_src])
    exe.assign_output_artifacts([d_out])
    print("lineage execution:", exe.resource_name)
    print("  inputs:", [a.display_name for a in exe.get_input_artifacts()])
    print("  outputs:", [a.display_name for a in exe.get_output_artifacts()])
except Exception as e:
    print("execution err:", repr(e)[:200])
