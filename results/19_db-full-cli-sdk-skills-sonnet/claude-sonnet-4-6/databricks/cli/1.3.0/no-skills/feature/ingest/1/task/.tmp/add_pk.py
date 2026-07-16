# Add primary key constraint to the feature table
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Add primary key constraint (required for Feature Engineering in UC)
spark.sql("""
    ALTER TABLE workspace.mlpabcbef07.transactions9dd1da
    ADD CONSTRAINT transactions9dd1da_pk PRIMARY KEY (row_id)
""")

print("Primary key constraint added successfully")

# Verify
result = spark.sql("DESCRIBE TABLE EXTENDED workspace.mlpabcbef07.transactions9dd1da")
result.show(50, truncate=False)
