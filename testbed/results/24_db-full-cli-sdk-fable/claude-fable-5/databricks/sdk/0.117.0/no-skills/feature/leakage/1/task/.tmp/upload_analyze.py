import os
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpabd74aad
catalog, sch = schema.split(".")
wh_id = "a832b544eb7dc3fe"

def sql(stmt):
    r = w.statement_execution.execute_statement(statement=stmt, warehouse_id=wh_id, wait_timeout="50s")
    if r.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"{r.status.state}: {r.status.error}")
    return r

# volume for the csv
sql(f"CREATE VOLUME IF NOT EXISTS {schema}.raw")
with open("data/training_data.csv", "rb") as f:
    w.files.upload(f"/Volumes/{catalog}/{sch}/raw/training_data.csv", f, overwrite=True)

sql(f"""
CREATE OR REPLACE TABLE {schema}.training_data AS
SELECT row_id, f1::double f1, f2::double f2, f3::double f3, f4::double f4, f5::double f5, f6::double f6, label::int label
FROM read_files('/Volumes/{catalog}/{sch}/raw/training_data.csv', format => 'csv', header => true)
""")

r = sql(f"SELECT count(*), sum(label) FROM {schema}.training_data")
print("rows, positives:", r.result.data_array[0])
