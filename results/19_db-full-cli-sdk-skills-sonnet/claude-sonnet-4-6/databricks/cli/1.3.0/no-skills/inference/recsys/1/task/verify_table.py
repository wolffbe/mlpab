# Databricks notebook source
# COMMAND ----------
# Verify the feature table data

table_name = "workspace.mlpabb40f43.recs708df6"

# Check schema
print("Schema:")
spark.table(table_name).printSchema()

# Check total rows
total = spark.sql(f"SELECT COUNT(*) as count FROM {table_name}").collect()[0]["count"]
print(f"\nTotal rows: {total}")

# Check user count and rows per user
user_counts = spark.sql(f"""
SELECT user_id, COUNT(*) as num_recs, MIN(rank) as min_rank, MAX(rank) as max_rank
FROM {table_name}
GROUP BY user_id
ORDER BY user_id
""")
print(f"\nUsers with recommendations:")
user_counts.show(50, truncate=False)

# Verify rec_id format
print("\nSample data:")
spark.sql(f"SELECT * FROM {table_name} ORDER BY user_id, rank LIMIT 20").show(20, truncate=False)

# Check for any violations
distinct_users = spark.sql(f"SELECT COUNT(DISTINCT user_id) as n_users FROM {table_name}").collect()[0]["n_users"]
print(f"\nDistinct users: {distinct_users}")

# Verify format of rec_id
bad_rec_ids = spark.sql(f"""
SELECT rec_id
FROM {table_name}
WHERE rec_id NOT RLIKE '^U[0-9]{{4}}#[1-5]$'
""").count()
print(f"Bad rec_id format count: {bad_rec_ids}")

# Verify rank is 1-5
bad_ranks = spark.sql(f"SELECT COUNT(*) as n FROM {table_name} WHERE rank < 1 OR rank > 5").collect()[0]["n"]
print(f"Bad rank values: {bad_ranks}")

result = {
    "total_rows": total,
    "distinct_users": distinct_users,
    "expected_rows": 200,
    "bad_rec_ids": bad_rec_ids,
    "bad_ranks": bad_ranks
}

import json
dbutils.notebook.exit(json.dumps(result))
