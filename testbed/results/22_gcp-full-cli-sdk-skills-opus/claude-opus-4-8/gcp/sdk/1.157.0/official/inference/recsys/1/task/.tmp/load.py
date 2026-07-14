import os
import google.cloud.bigquery as bigquery

proj = os.environ['GCP_PROJECT']
ds = os.environ['GCP_BQ_DATASET']
c = bigquery.Client(project=proj)

def load(local_path, table):
    tid = "{}.{}.{}".format(proj, ds, table)
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition="WRITE_TRUNCATE",
    )
    with open(local_path, "rb") as f:
        job = c.load_table_from_file(f, tid, job_config=job_config)
    job.result()
    t = c.get_table(tid)
    print("loaded {} rows into {} schema={}".format(t.num_rows, table,
          [(s.name, s.field_type) for s in t.schema]))

load("data/user_embeddings.csv", "stg_user_emb")
load("data/item_embeddings.csv", "stg_item_emb")
load("data/interactions.csv", "stg_interactions")
