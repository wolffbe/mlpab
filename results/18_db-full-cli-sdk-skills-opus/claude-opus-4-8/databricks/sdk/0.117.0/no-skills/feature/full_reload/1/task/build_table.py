from databricks.sdk import WorkspaceClient
import os, time

w = WorkspaceClient()
WH = '4dfab06c923fe3cc'
cat, sch = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
TBL = 'customers302d18'
FQ = "`%s`.`%s`.`%s`" % (cat, sch, TBL)
VOL = "/Volumes/%s/%s/data_vol" % (cat, sch)


def runsql(sql, fetch=False):
    r = w.statement_execution.execute_statement(statement=sql, warehouse_id=WH, wait_timeout='50s')
    sid = r.statement_id
    st = r.status.state.value
    while st in ('PENDING', 'RUNNING'):
        time.sleep(2)
        r = w.statement_execution.get_statement(sid)
        st = r.status.state.value
    if st != 'SUCCEEDED':
        raise RuntimeError("SQL FAILED: %s :: %s\n%s" % (st, r.status.error, sql[:200]))
    if fetch and r.result and r.result.data_array is not None:
        return r.result.data_array
    return None


# ---------- VERSION 1: initial schema ----------
runsql("DROP TABLE IF EXISTS %s" % FQ)
runsql(
    "CREATE TABLE %s ("
    "  row_id STRING NOT NULL,"
    "  name STRING,"
    "  balance_eur DOUBLE,"
    "  updated_at BIGINT NOT NULL,"
    "  CONSTRAINT pk_v1 PRIMARY KEY (row_id, updated_at TIMESERIES)"
    ") TBLPROPERTIES (delta.enableChangeDataFeed = true)" % FQ
)
runsql(
    "COPY INTO %s FROM ("
    "  SELECT row_id::string AS row_id, name::string AS name,"
    "         balance_eur::double AS balance_eur, updated_at::bigint AS updated_at"
    "  FROM '%s/initial_export.csv'"
    ") FILEFORMAT = CSV "
    "FORMAT_OPTIONS ('header'='true', 'inferSchema'='false') "
    "COPY_OPTIONS ('mergeSchema'='false')" % (FQ, VOL)
)
n1 = runsql("SELECT count(*), count(distinct row_id) FROM %s" % FQ, fetch=True)
print("V1 loaded rows / distinct row_id:", n1)
hist1 = runsql("DESCRIBE HISTORY %s LIMIT 3" % FQ, fetch=True)
print("V1 latest delta version:", hist1[0][0] if hist1 else None)

# ---------- VERSION 2: breaking new schema, full reload from scratch ----------
runsql(
    "CREATE OR REPLACE TABLE %s ("
    "  row_id STRING NOT NULL,"
    "  full_name STRING,"
    "  balance DOUBLE,"
    "  currency STRING,"
    "  updated_at BIGINT NOT NULL,"
    "  CONSTRAINT pk_v2 PRIMARY KEY (row_id, updated_at TIMESERIES)"
    ") TBLPROPERTIES (delta.enableChangeDataFeed = true)" % FQ
)
runsql(
    "COPY INTO %s FROM ("
    "  SELECT row_id::string AS row_id, full_name::string AS full_name,"
    "         balance::double AS balance, currency::string AS currency,"
    "         updated_at::bigint AS updated_at"
    "  FROM '%s/new_export.csv'"
    ") FILEFORMAT = CSV "
    "FORMAT_OPTIONS ('header'='true', 'inferSchema'='false') "
    "COPY_OPTIONS ('mergeSchema'='false')" % (FQ, VOL)
)
n2 = runsql("SELECT count(*), count(distinct row_id) FROM %s" % FQ, fetch=True)
print("V2 loaded rows / distinct row_id:", n2)
cols = runsql("SELECT * FROM %s LIMIT 2" % FQ, fetch=True)
print("V2 sample:", cols)
desc = runsql("DESCRIBE TABLE %s" % FQ, fetch=True)
print("V2 columns:", [r[0] for r in desc if r[0] and not r[0].startswith('#')])
hist2 = runsql("DESCRIBE HISTORY %s" % FQ, fetch=True)
print("V2 delta versions present:", [r[0] for r in hist2])
