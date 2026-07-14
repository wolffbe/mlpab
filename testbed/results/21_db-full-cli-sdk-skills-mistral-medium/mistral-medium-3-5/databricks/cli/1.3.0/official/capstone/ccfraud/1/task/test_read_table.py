print("Reading table...")

catalog, schema = 'workspace', 'mlpabfcf9c1'

spark_df = spark.table(f"{catalog}.{schema}.cctxn015310")
print(f"Read {spark_df.count()} rows")
print(f"Columns: {spark_df.columns}")
print("Done")
