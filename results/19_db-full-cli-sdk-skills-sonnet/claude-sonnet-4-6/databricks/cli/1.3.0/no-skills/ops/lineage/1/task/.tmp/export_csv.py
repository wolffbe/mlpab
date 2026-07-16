df = spark.sql("SELECT row_id, col_sum FROM workspace.mlpabbb38f1.derivedd05474 ORDER BY row_id")
df.coalesce(1).write.mode("overwrite").option("header", "true").csv("/Volumes/workspace/mlpabbb38f1/mlpabbb38f1_vol/derivedd05474_export")
files = dbutils.fs.ls("/Volumes/workspace/mlpabbb38f1/mlpabbb38f1_vol/derivedd05474_export")
csv_files = [f.path for f in files if f.path.endswith('.csv')]
dbutils.notebook.exit(str(csv_files))
