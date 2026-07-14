from pyspark.sql import SparkSession
from pyspark.sql.functions import corr

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

# Define the schema and table name
schema = "${MLPAB_DATABRICKS_SCHEMA}"
table_name = "${MLPAB_DATABRICKS_PREFIX}_training_data"

# Read the table
df = spark.table(f"{schema}.{table_name}")

# Compute correlation of each feature with the label
features = ["f1", "f2", "f3", "f4", "f5", "f6"]
correlations = {}

for feature in features:
    correlation = df.select(corr(feature, "label")).collect()[0][0]
    correlations[feature] = correlation

# Identify the feature with the highest absolute correlation
leaking_feature = max(correlations.items(), key=lambda x: abs(x[1]))[0]

# Print the result
print(f"Leaking feature: {leaking_feature}")
print("Correlations:")
for feature, correlation in correlations.items():
    print(f"{feature}: {correlation}")