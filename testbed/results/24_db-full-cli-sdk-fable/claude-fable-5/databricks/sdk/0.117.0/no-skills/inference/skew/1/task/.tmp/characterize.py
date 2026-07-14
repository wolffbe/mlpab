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


q = f"""
WITH j AS (
  SELECT t.entity_id, t.f2 tf2, s.f2 sf2
  FROM {SCHEMA}.skew_training t JOIN {SCHEMA}.skew_serving s USING (entity_id)
)
SELECT avg(tf2), stddev(tf2), avg(sf2), stddev(sf2),
       avg(sf2 / tf2), stddev(sf2 / tf2),
       avg(sf2 - tf2), stddev(sf2 - tf2),
       corr(log(tf2), log(sf2)),
       avg(sf2 - tf2*tf2), avg(sf2 - sqrt(tf2)), avg(sf2 - exp(tf2/10)),
       avg(log(sf2) / log(tf2))
FROM j
"""
r = sql(q)
for c, v in zip(r.manifest.schema.columns, r.result.data_array[0]):
    print(c.name, "=", v)

r = sql(f"""
SELECT t.entity_id, t.f2, s.f2
FROM {SCHEMA}.skew_training t JOIN {SCHEMA}.skew_serving s USING (entity_id)
ORDER BY t.entity_id LIMIT 12
""")
for row in r.result.data_array:
    print(row)
