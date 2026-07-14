from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import ExecuteStatementRequest, StatementResponse
import time

# Initialize the Workspace Client
w = WorkspaceClient()

# SQL query to generate the training dataset
query = """
CREATE OR REPLACE TABLE workspace.mlpab674210.churntrainingaf8b21 AS
SELECT 
    l.account_id,
    l.label_time,
    t.amount,
    t.balance,
    p.credit_score,
    p.tier,
    a.sessions_7d,
    h.health_score,
    l.churned
FROM labels l
LEFT JOIN (
    SELECT 
        t.account_id,
        t.event_time,
        t.amount,
        t.balance,
        l.label_time
    FROM labels l
    JOIN transactions t ON t.account_id = l.account_id AND t.event_time <= l.label_time
    QUALIFY ROW_NUMBER() OVER (PARTITION BY t.account_id, l.label_time ORDER BY t.event_time DESC) = 1
) t ON l.account_id = t.account_id AND l.label_time = t.label_time
LEFT JOIN (
    SELECT 
        p.account_id,
        p.event_time,
        p.credit_score,
        p.tier,
        l.label_time
    FROM labels l
    JOIN profiles p ON p.account_id = l.account_id AND p.event_time <= l.label_time
    QUALIFY ROW_NUMBER() OVER (PARTITION BY p.account_id, l.label_time ORDER BY p.event_time DESC) = 1
) p ON l.account_id = p.account_id AND l.label_time = p.label_time
LEFT JOIN (
    SELECT 
        a.account_id,
        a.event_time,
        a.sessions_7d,
        l.label_time
    FROM labels l
    JOIN activity a ON a.account_id = l.account_id AND a.event_time <= l.label_time
    QUALIFY ROW_NUMBER() OVER (PARTITION BY a.account_id, l.label_time ORDER BY a.event_time DESC) = 1
) a ON l.account_id = a.account_id AND l.label_time = a.label_time
LEFT JOIN (
    SELECT 
        h.account_id,
        h.event_time,
        h.health_score,
        l.label_time
    FROM labels l
    JOIN account_health h ON h.account_id = l.account_id AND h.event_time <= l.label_time
    QUALIFY ROW_NUMBER() OVER (PARTITION BY h.account_id, l.label_time ORDER BY h.event_time DESC) = 1
) h ON l.account_id = h.account_id AND l.label_time = h.label_time
"""

# Execute the query
print("Executing SQL query...")
try:
    # Submit the query
    statement = w.statement_execution.execute_statement(
        warehouse_id="e0610a1529543492b",  # Serverless SQL warehouse
        catalog="workspace",
        schema="mlpab674210",
        statement=query
    )
    
    # Poll for completion
    while True:
        result = w.statement_execution.get_statement(statement.statement_id)
        if result.status.state in ["SUCCEEDED", "FAILED", "CANCELED"]:
            break
        time.sleep(5)
    
    if result.status.state == "SUCCEEDED":
        print("Query succeeded. Training dataset created.")
    else:
        print(f"Query failed: {result.status.error_message}")
except Exception as e:
    print(f"Error: {e}")