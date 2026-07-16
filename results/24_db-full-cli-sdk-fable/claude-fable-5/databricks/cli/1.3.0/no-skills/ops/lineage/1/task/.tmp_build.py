# Databricks notebook source
stmts = [
    """CREATE OR REPLACE TABLE workspace.mlpabf9499e.rawa5a60b6 AS
     SELECT row_id, CAST(a_val AS DOUBLE) AS a_val
     FROM read_files('/Volumes/workspace/mlpabf9499e/raw_data/raw_a.csv', format => 'csv', header => true)""",
    "ALTER TABLE workspace.mlpabf9499e.rawa5a60b6 ALTER COLUMN row_id SET NOT NULL",
    "ALTER TABLE workspace.mlpabf9499e.rawa5a60b6 ADD CONSTRAINT rawa5a60b6_pk PRIMARY KEY (row_id)",
    """CREATE OR REPLACE TABLE workspace.mlpabf9499e.rawb5a60b6 AS
     SELECT row_id, CAST(b_val AS DOUBLE) AS b_val
     FROM read_files('/Volumes/workspace/mlpabf9499e/raw_data/raw_b.csv', format => 'csv', header => true)""",
    "ALTER TABLE workspace.mlpabf9499e.rawb5a60b6 ALTER COLUMN row_id SET NOT NULL",
    "ALTER TABLE workspace.mlpabf9499e.rawb5a60b6 ADD CONSTRAINT rawb5a60b6_pk PRIMARY KEY (row_id)",
    """CREATE OR REPLACE TABLE workspace.mlpabf9499e.derived5a60b6
     TBLPROPERTIES (delta.enableChangeDataFeed = true) AS
     SELECT a.row_id, ROUND(a.a_val + b.b_val, 6) AS col_sum
     FROM workspace.mlpabf9499e.rawa5a60b6 a
     JOIN workspace.mlpabf9499e.rawb5a60b6 b ON a.row_id = b.row_id""",
    "ALTER TABLE workspace.mlpabf9499e.derived5a60b6 ALTER COLUMN row_id SET NOT NULL",
    "ALTER TABLE workspace.mlpabf9499e.derived5a60b6 ADD CONSTRAINT derived5a60b6_pk PRIMARY KEY (row_id)",
]
for s in stmts:
    print(s.split("\n")[0])
    spark.sql(s)
print(
    "counts:",
    spark.sql(
        "SELECT (SELECT count(*) FROM workspace.mlpabf9499e.rawa5a60b6) a,"
        " (SELECT count(*) FROM workspace.mlpabf9499e.rawb5a60b6) b,"
        " (SELECT count(*) FROM workspace.mlpabf9499e.derived5a60b6) d"
    ).collect(),
)
