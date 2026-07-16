import csv, os, sys, time
import databricks.sdk
from databricks.sdk.service.sql import StatementState

w = databricks.sdk.WorkspaceClient()
WH = "4dfab06c923fe3cc"
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpab2e8eb1
CAT, SCH = SCHEMA.split(".")
FQ = lambda t: f"`{CAT}`.`{SCH}`.`{t}`"

def run(sql, label=""):
    r = w.statement_execution.execute_statement(
        warehouse_id=WH, statement=sql, wait_timeout="50s")
    sid = r.statement_id
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(sid)
    if r.status.state != StatementState.SUCCEEDED:
        print("FAILED", label, r.status.state, r.status.error)
        sys.exit(1)
    print("OK", label or sql[:60])
    return r

def read_csv(path):
    with open(path) as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = [row for row in rd]
    return header, rows

# ---- 1. Load embeddings + interactions into platform tables ----
ecols = "e1 DOUBLE, e2 DOUBLE, e3 DOUBLE, e4 DOUBLE, e5 DOUBLE, e6 DOUBLE, e7 DOUBLE, e8 DOUBLE"

run(f"CREATE OR REPLACE TABLE {FQ('user_emb')} (user_id STRING, {ecols})", "create user_emb")
run(f"CREATE OR REPLACE TABLE {FQ('item_emb')} (item_id STRING, {ecols})", "create item_emb")
run(f"CREATE OR REPLACE TABLE {FQ('interactions')} (user_id STRING, item_id STRING)", "create interactions")

def insert_batch(table, rows, n_num, idcol_count):
    vals = []
    for row in rows:
        ids = ",".join("'" + row[i].replace("'", "''") + "'" for i in range(idcol_count))
        nums = ",".join(row[idcol_count + j] for j in range(n_num))
        vals.append(f"({ids}{',' if nums else ''}{nums})")
    B = 100
    for k in range(0, len(vals), B):
        chunk = ",".join(vals[k:k+B])
        run(f"INSERT INTO {FQ(table)} VALUES {chunk}", f"insert {table} {k}")

uh, urows = read_csv("data/user_embeddings.csv")
insert_batch("user_emb", urows, 8, 1)
ih, irows = read_csv("data/item_embeddings.csv")
insert_batch("item_emb", irows, 8, 1)
xh, xrows = read_csv("data/interactions.csv")
insert_batch("interactions", xrows, 0, 2)

for t in ("user_emb", "item_emb", "interactions"):
    r = run(f"SELECT count(*) FROM {FQ(t)}", f"count {t}")
    print("  ", t, r.result.data_array[0][0])

# ---- 2. Compute top-5 recs entirely in SQL ----
dot = " + ".join(f"u.e{i}*i.e{i}" for i in range(1, 9))
recs_sql = f"""
CREATE OR REPLACE TABLE {FQ('recs2ead15')}
TBLPROPERTIES (delta.enableChangeDataFeed = true) AS
WITH scored AS (
  SELECT u.user_id, i.item_id, ({dot}) AS score
  FROM {FQ('user_emb')} u
  CROSS JOIN {FQ('item_emb')} i
  WHERE NOT EXISTS (
    SELECT 1 FROM {FQ('interactions')} x
    WHERE x.user_id = u.user_id AND x.item_id = i.item_id)
),
ranked AS (
  SELECT user_id, item_id, score,
         ROW_NUMBER() OVER (
           PARTITION BY user_id ORDER BY score DESC, item_id ASC) AS rnk
  FROM scored
)
SELECT concat(user_id, '#', CAST(rnk AS INT)) AS rec_id,
       user_id,
       CAST(rnk AS INT) AS rank,
       item_id
FROM ranked
WHERE rnk <= 5
"""
run(recs_sql, "build recs2ead15")

# ---- 3. Make it a feature table: NOT NULL + PRIMARY KEY on rec_id ----
run(f"ALTER TABLE {FQ('recs2ead15')} ALTER COLUMN rec_id SET NOT NULL", "set not null")
run(f"ALTER TABLE {FQ('recs2ead15')} ADD CONSTRAINT recs2ead15_pk PRIMARY KEY (rec_id)", "add pk")

r = run(f"SELECT count(*) FROM {FQ('recs2ead15')}", "count recs")
print("TOTAL RECS:", r.result.data_array[0][0])
r = run(f"SELECT * FROM {FQ('recs2ead15')} WHERE user_id='U0003' ORDER BY rank", "sample U0003")
for row in r.result.data_array:
    print("   ", row)
print("DONE_SQL")
