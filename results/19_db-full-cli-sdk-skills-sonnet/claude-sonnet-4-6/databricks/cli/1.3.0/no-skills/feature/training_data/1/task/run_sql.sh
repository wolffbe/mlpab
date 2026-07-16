#!/bin/bash
# run_sql.sh - Execute a SQL statement via Databricks Statement Execution API
# Usage: ./run_sql.sh "SQL STATEMENT"

WAREHOUSE_ID="4dfab06c923fe3cc"
STATEMENT="$1"

# Write statement to temp file to avoid quoting issues
TMPFILE=$(mktemp /tmp/sql_stmt_XXXXXX.json)
cat > "$TMPFILE" << ENDJSON
{
  "warehouse_id": "$WAREHOUSE_ID",
  "statement": $(echo "$STATEMENT" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))"),
  "wait_timeout": "50s",
  "on_wait_timeout": "CANCEL"
}
ENDJSON

echo "Executing SQL..." >&2
RESPONSE=$(databricks api post /api/2.0/sql/statements --body "@$TMPFILE" -o json 2>&1)
rm -f "$TMPFILE"

STMT_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('statement_id',''))" 2>/dev/null)
STATUS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',{}).get('state',''))" 2>/dev/null)

echo "Statement ID: $STMT_ID, Status: $STATUS" >&2

# Poll if still running
MAX_WAIT=300
ELAPSED=0
while [ "$STATUS" = "RUNNING" ] || [ "$STATUS" = "PENDING" ]; do
  sleep 5
  ELAPSED=$((ELAPSED + 5))
  if [ $ELAPSED -gt $MAX_WAIT ]; then
    echo "TIMEOUT waiting for statement $STMT_ID" >&2
    break
  fi
  RESPONSE=$(databricks api get /api/2.0/sql/statements/$STMT_ID -o json 2>&1)
  STATUS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',{}).get('state',''))" 2>/dev/null)
  echo "  Status: $STATUS (${ELAPSED}s)" >&2
done

if [ "$STATUS" = "SUCCEEDED" ]; then
  echo "SUCCESS" >&2
  echo "$RESPONSE"
else
  echo "FAILED with status: $STATUS" >&2
  echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get('status',{}), indent=2))" 2>/dev/null >&2
  echo "$RESPONSE"
  exit 1
fi
