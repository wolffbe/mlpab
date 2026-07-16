#!/usr/bin/env python3
"""Check catalog configuration and table details."""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
CATALOG, SCHEMA_NAME = SCHEMA.split(".", 1)
TABLE_NAME = "eventsd693d3"
FULL_TABLE = f"{CATALOG}.{SCHEMA_NAME}.{TABLE_NAME}"

w = WorkspaceClient()

# Get warehouse
warehouses = list(w.warehouses.list())
wh_id = warehouses[0].id

def exec_sql(sql):
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=wh_id,
        wait_timeout="30s",
    )
    if resp.status.state == StatementState.SUCCEEDED:
        return resp.result
    raise RuntimeError(f"SQL failed: {resp.status.error}")

# Check catalogs
print("Available catalogs:")
r = exec_sql("SHOW CATALOGS")
if r and r.data_array:
    for row in r.data_array:
        print(f"  {row}")

# Check the specific catalog
print(f"\nCatalog '{CATALOG}' details:")
try:
    cat = w.catalogs.get(CATALOG)
    print(f"  Name: {cat.name}, Type: {cat.catalog_type}")
except Exception as e:
    print(f"  Error: {e}")

# Verify our table exists
print(f"\nTable {FULL_TABLE}:")
try:
    tbl = w.tables.get(FULL_TABLE)
    print(f"  Name: {tbl.name}, Type: {tbl.table_type}")
    print(f"  Full name: {tbl.full_name}")
    print(f"  Catalog: {tbl.catalog_name}")
    print(f"  Schema: {tbl.schema_name}")
except Exception as e:
    print(f"  Error: {e}")

# Try feature store get table
print(f"\nFeature store get table {FULL_TABLE}:")
try:
    result = w.api_client.do("GET", f"/api/2.0/feature-store/tables/{FULL_TABLE}")
    print(f"  Result: {result}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")
