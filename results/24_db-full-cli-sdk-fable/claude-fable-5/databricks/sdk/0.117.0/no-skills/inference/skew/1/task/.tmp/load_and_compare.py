import databricks.sdk

w = databricks.sdk.WorkspaceClient()
WH = "8a93fc195da2ceb1"
SCHEMA = "workspace.mlpaba23cda"


def sql(stmt):
    r = w.statement_execution.execute_statement(
        statement=stmt, warehouse_id=WH, wait_timeout="50s"
    )
    assert r.status.state.value == "SUCCEEDED", (r.status.state, r.status.error)
    return r


def load(path, table):
    lines = open(path).read().strip().split("\n")
    vals = []
    for line in lines[1:]:
        parts = line.split(",")
        vals.append("('%s',%s)" % (parts[0], ",".join(parts[1:])))
    sql(
        f"CREATE OR REPLACE TABLE {SCHEMA}.{table} "
        "(entity_id STRING, f1 DOUBLE, f2 DOUBLE, f3 DOUBLE, f4 DOUBLE, f5 DOUBLE)"
    )
    sql(f"INSERT INTO {SCHEMA}.{table} VALUES {','.join(vals)}")
    print("loaded", table, len(vals))


load("data/training_sample.csv", "skew_training")
load("data/serving_log.csv", "skew_serving")

# Compare per-feature: join on entity_id, measure divergence between paths
query = f"""
WITH j AS (
  SELECT t.entity_id,
         t.f1 tf1, t.f2 tf2, t.f3 tf3, t.f4 tf4, t.f5 tf5,
         s.f1 sf1, s.f2 sf2, s.f3 sf3, s.f4 sf4, s.f5 sf5
  FROM {SCHEMA}.skew_training t
  JOIN {SCHEMA}.skew_serving s USING (entity_id)
)
SELECT
  count(*) AS n,
  avg(abs(tf1 - sf1)) AS mad_f1, avg(abs(tf2 - sf2)) AS mad_f2,
  avg(abs(tf3 - sf3)) AS mad_f3, avg(abs(tf4 - sf4)) AS mad_f4,
  avg(abs(tf5 - sf5)) AS mad_f5,
  corr(tf1, sf1) AS c1, corr(tf2, sf2) AS c2, corr(tf3, sf3) AS c3,
  corr(tf4, sf4) AS c4, corr(tf5, sf5) AS c5
FROM j
"""
r = sql(query)
cols = [c.name for c in r.manifest.schema.columns]
row = r.result.data_array[0]
for c, v in zip(cols, row):
    print(c, "=", v)
