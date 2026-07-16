from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
import time

w = WorkspaceClient()
warehouse_id = '4dfab06c923fe3cc'

sql = "COPY INTO workspace.mlpab913631.prediction_log FROM '/Volumes/workspace/mlpab913631/prediction_data/prediction_log.csv' FILEFORMAT = CSV FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')"

stmt = w.statement_execution.execute_statement(
    statement=sql,
    warehouse_id=warehouse_id,
    catalog='workspace',
    schema='mlpab913631',
    wait_timeout='0s'
)
stmt_id = stmt.statement_id
print('Statement ID:', stmt_id)

for i in range(30):
    stmt = w.statement_execution.get_statement(stmt_id)
    print(f'Attempt {i+1}: {stmt.status.state}')
    if stmt.status.state in [StatementState.SUCCEEDED, StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED]:
        break
    time.sleep(3)

print('Final status:', stmt.status.state)
if stmt.status.error:
    print('Error:', stmt.status.error)
if stmt.result and stmt.result.data_array:
    print('Result:', stmt.result.data_array)
