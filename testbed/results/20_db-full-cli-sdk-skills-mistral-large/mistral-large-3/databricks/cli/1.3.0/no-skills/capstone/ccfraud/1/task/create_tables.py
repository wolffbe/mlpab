from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql

w = WorkspaceClient()

# Create transactions table
w.statement_execution.execute_statement(
    warehouse_id="4dfab06c923fe3cc",
    catalog="workspace",
    schema="${MLPAB_DATABRICKS_SCHEMA}",
    statement="CREATE TABLE transactions USING CSV OPTIONS (path '/dbfs${PWD}/data/transactions.csv', header 'true', inferSchema 'true')"
)

# Create score_transactions table
w.statement_execution.execute_statement(
    warehouse_id="4dfab06c923fe3cc",
    catalog="workspace",
    schema="${MLPAB_DATABRICKS_SCHEMA}",
    statement="CREATE TABLE score_transactions USING CSV OPTIONS (path '/dbfs${PWD}/data/score_transactions.csv', header 'true', inferSchema 'true')"
)

# Verify tables
result = w.statement_execution.execute_statement(
    warehouse_id="4dfab06c923fe3cc",
    catalog="workspace",
    schema="${MLPAB_DATABRICKS_SCHEMA}",
    statement="SHOW TABLES"
)

print(result)