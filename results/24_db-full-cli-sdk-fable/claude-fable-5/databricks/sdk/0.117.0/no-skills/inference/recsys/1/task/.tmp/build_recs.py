import csv, os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpab00e5ae
TABLE = f"{SCHEMA}.recs9e4a4c"

wh = next(x for x in w.warehouses.list() if x.name == "Serverless Starter Warehouse")
print("using warehouse:", wh.id, wh.name, flush=True)

def run(sql, max_wait=600):
    r = w.statement_execution.execute_statement(
        statement=sql, warehouse_id=wh.id, wait_timeout="50s")
    waited = 0
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        if waited > max_wait:
            raise RuntimeError("statement stuck > max_wait")
        time.sleep(3); waited += 3
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed: {r.status.state} {r.status.error}")
    return r

def rows(path):
    with open(path) as f:
        return list(csv.reader(f))

emb_cols = ",".join(f"e{i}" for i in range(1, 9))

def values_emb(path, key):
    data = rows(path)[1:]
    return ",\n".join("('" + r[0] + "'," + ",".join(r[1:]) + ")" for r in data)

users_v = values_emb("data/user_embeddings.csv", "user_id")
items_v = values_emb("data/item_embeddings.csv", "item_id")
inter_v = ",\n".join(f"('{u}','{i}')" for u, i in rows("data/interactions.csv")[1:])

run(f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
  rec_id STRING NOT NULL,
  user_id STRING,
  `rank` INT,
  item_id STRING,
  CONSTRAINT recs9e4a4c_pk PRIMARY KEY (rec_id)
)
COMMENT 'Feature table recs9e4a4c version 1: top-5 two-tower dot-product recommendations per user'
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
print("table created")

dot = "+".join(f"u.e{i}*i.e{i}" for i in range(1, 9))
run(f"""
INSERT OVERWRITE {TABLE}
WITH users({("user_id," + emb_cols)}) AS (VALUES {users_v}),
items({("item_id," + emb_cols)}) AS (VALUES {items_v}),
inter(user_id, item_id) AS (VALUES {inter_v}),
scored AS (
  SELECT u.user_id, i.item_id, {dot} AS score
  FROM users u CROSS JOIN items i
  WHERE NOT EXISTS (
    SELECT 1 FROM inter t WHERE t.user_id = u.user_id AND t.item_id = i.item_id)
),
ranked AS (
  SELECT user_id, item_id,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY score DESC, item_id ASC) AS rnk
  FROM scored
)
SELECT CONCAT(user_id, '#', rnk) AS rec_id, user_id, CAST(rnk AS INT) AS `rank`, item_id
FROM ranked WHERE rnk <= 5
""")
print("data inserted")

r = run(f"SELECT COUNT(*) c, COUNT(DISTINCT user_id) u FROM {TABLE}")
print("rows,users:", r.result.data_array)
r = run(f"SELECT * FROM {TABLE} WHERE user_id='U0003' ORDER BY `rank`")
print("sample U0003:", r.result.data_array)
