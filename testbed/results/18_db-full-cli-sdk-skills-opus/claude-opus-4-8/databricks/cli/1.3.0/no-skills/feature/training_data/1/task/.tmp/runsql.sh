#!/bin/zsh
# Usage: runsql.sh "<SQL>"
WID="4dfab06c923fe3cc"
SQL="$1"
PAYLOAD=$(mktemp)
RESP=$(mktemp)
# build json payload safely via python-free jq-less heredoc using databricks json passthrough
python3 - "$SQL" "$WID" > "$PAYLOAD" <<'PYEOF'
import json,sys
print(json.dumps({"warehouse_id":sys.argv[2],"statement":sys.argv[1],"wait_timeout":"50s","on_wait_timeout":"CONTINUE","format":"JSON_ARRAY","disposition":"INLINE"}))
PYEOF
databricks api post /api/2.0/sql/statements --json @"$PAYLOAD" > "$RESP" 2>&1
cat "$RESP"
