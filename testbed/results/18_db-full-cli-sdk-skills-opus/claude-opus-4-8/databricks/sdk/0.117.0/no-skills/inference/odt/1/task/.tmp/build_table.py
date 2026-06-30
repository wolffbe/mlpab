from databricks.sdk import WorkspaceClient
import os

w = WorkspaceClient()
catalog, schema = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
wh = '4dfab06c923fe3cc'
tbl = f'{catalog}.{schema}.scoredbfc4ef'
base = f'/Volumes/{catalog}/{schema}/src_data'


def sql(s):
    r = w.statement_execution.execute_statement(statement=s, warehouse_id=wh, wait_timeout='50s')
    st = r.status.state.value
    if st != 'SUCCEEDED':
        raise RuntimeError(f'{st}: {getattr(r.status, "error", None)} :: {s[:120]}')
    return r


sql(f'DROP TABLE IF EXISTS {tbl}')

create = (
    f'CREATE TABLE {tbl} ('
    'request_id STRING NOT NULL, account_id STRING, distance_deg DOUBLE, score DOUBLE, '
    'CONSTRAINT scoredbfc4ef_pk PRIMARY KEY(request_id)) '
    'TBLPROPERTIES (delta.enableChangeDataFeed = true)'
)
sql(create)
print('table created')

insert = (
    f"INSERT INTO {tbl} "
    "WITH j AS (SELECT r.request_id, r.account_id, "
    "round(sqrt(pow(r.request_lat - p.home_lat,2) + pow(r.request_lon - p.home_lon,2)),6) AS distance_deg, "
    "p.base_score "
    f"FROM read_files('{base}/requests.csv', format=>'csv', header=>true) r "
    f"JOIN read_files('{base}/profiles.csv', format=>'csv', header=>true) p "
    "ON r.account_id = p.account_id) "
    "SELECT request_id, account_id, distance_deg, round(base_score - 0.1*distance_deg, 6) AS score FROM j"
)
sql(insert)
print('data inserted')

r = sql(f'SELECT count(*) c, count(distinct request_id) d FROM {tbl}')
print('counts:', r.result.data_array)
r = sql(f'SELECT * FROM {tbl} ORDER BY request_id LIMIT 3')
print('sample:', r.result.data_array)
