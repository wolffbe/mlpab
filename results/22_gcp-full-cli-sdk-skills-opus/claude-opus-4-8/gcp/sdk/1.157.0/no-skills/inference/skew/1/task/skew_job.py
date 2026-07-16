import os
from google.cloud import bigquery

project = os.environ["GCP_PROJECT"]
dataset = os.environ["GCP_BQ_DATASET"]
client = bigquery.Client(project=project)

schema = [
    bigquery.SchemaField("entity_id", "STRING"),
    bigquery.SchemaField("f1", "FLOAT"),
    bigquery.SchemaField("f2", "FLOAT"),
    bigquery.SchemaField("f3", "FLOAT"),
    bigquery.SchemaField("f4", "FLOAT"),
    bigquery.SchemaField("f5", "FLOAT"),
]


def load(tbl, path):
    tid = f"{project}.{dataset}.{tbl}"
    job_config = bigquery.LoadJobConfig(
        schema=schema, skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
        write_disposition="WRITE_TRUNCATE",
    )
    with open(path, "rb") as f:
        job = client.load_table_from_file(f, tid, job_config=job_config)
    job.result()
    print("loaded", tid, client.get_table(tid).num_rows, "rows")


load("skew_train", "data/training_sample.csv")
load("skew_serve", "data/serving_log.csv")

feats = ["f1", "f2", "f3", "f4", "f5"]
sel = ",\n".join([
    f"AVG(ABS(t.{f}-s.{f})) AS mad_{f}, "
    f"CORR(t.{f}, s.{f}) AS corr_{f}, "
    f"AVG(t.{f}) AS tmean_{f}, AVG(s.{f}) AS smean_{f}, "
    f"STDDEV(t.{f}) AS tstd_{f}, STDDEV(s.{f}) AS sstd_{f}, "
    f"AVG(ABS(t.{f}-s.{f})/NULLIF(ABS(t.{f}),0)) AS relmad_{f}"
    for f in feats
])
q = f"""
SELECT COUNT(*) AS n, {sel}
FROM `{project}.{dataset}.skew_train` t
JOIN `{project}.{dataset}.skew_serve` s USING(entity_id)
"""
row = list(client.query(q).result())[0]
print("matched entities:", row["n"])
print(f"{'feat':4} {'MAD':>10} {'relMAD':>10} {'CORR':>8} {'tmean':>8} {'smean':>8} {'tstd':>8} {'sstd':>8}")
for f in feats:
    print(f"{f:4} {row['mad_'+f]:>10.4f} {row['relmad_'+f]:>10.4f} {row['corr_'+f]:>8.4f} "
          f"{row['tmean_'+f]:>8.3f} {row['smean_'+f]:>8.3f} {row['tstd_'+f]:>8.3f} {row['sstd_'+f]:>8.3f}")
