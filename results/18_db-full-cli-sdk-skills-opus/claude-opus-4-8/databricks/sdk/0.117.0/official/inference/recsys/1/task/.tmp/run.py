import databricks.sdk
import sys

w = databricks.sdk.WorkspaceClient()
cat, sch = 'workspace', 'mlpab92a5cb'
wh = '4dfab06c923fe3cc'


def sql(q):
    r = w.statement_execution.execute_statement(
        warehouse_id=wh, statement=q, catalog=cat, schema=sch, wait_timeout='50s')
    st = r.status.state.value
    if st != 'SUCCEEDED':
        print('FAIL', st, r.status.error)
        raise SystemExit(1)
    return r


sql('CREATE VOLUME IF NOT EXISTS data_vol')
print('volume created')

base = f'/Volumes/{cat}/{sch}/data_vol'
for fn in ['interactions.csv', 'user_embeddings.csv', 'item_embeddings.csv']:
    with open('data/' + fn, 'rb') as f:
        w.files.upload(f'{base}/{fn}', f, overwrite=True)
    print('uploaded', fn)
print('DONE upload')
