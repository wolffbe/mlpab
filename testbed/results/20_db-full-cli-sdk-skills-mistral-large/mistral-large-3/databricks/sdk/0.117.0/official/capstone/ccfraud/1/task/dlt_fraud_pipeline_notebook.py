# Databricks notebook source
# MAGIC %md
# Fraud FTI Pipeline (DLT)

This notebook defines a Delta Live Table (DLT) pipeline for:
1. Feature engineering into `cctxne0b071`.
2. Training dataset `cctde0b071`.
3. Model `ccmodele0b071`.
4. Predictions table `ccprede0b071` with low-latency lookup.

---

# COMMAND ----------

import os
import dlt
from pyspark.sql.functions import col, unix_timestamp, lag, stddev, mean, count, lit, when
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# Config
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.mlpab21b96f
catalog_name = schema_name.split(".")[0]  # workspace
schema_name_only = schema_name.split(".")[1]  # mlpab21b96f
prefix = os.getenv("MLPAB_DATABRICKS_PREFIX")  # mlpab21b96f

# Feature Group, Dataset, Model, and Predictions Table Names
feature_group_name = "cctxne0b071"
training_dataset_name = "cctde0b071"
model_name = "ccmodele0b071"
predictions_table_name = "ccprede0b071"

# Paths
volume_path = f"/Volumes/{catalog_name}/{schema_name_only}/{prefix}_fraud_data"
transactions_path = f"{volume_path}/transactions.csv"
score_path = f"{volume_path}/score_transactions.csv"

# COMMAND ----------

@dlt.table(
    name=feature_group_name,
    comment="Fraud feature group: transaction velocity, amount stats, geo distance."
)
def create_feature_group():
    # Read transactions
    transactions_df = spark.read.csv(transactions_path, header=True, inferSchema=True)
    
    # Feature Engineering
    transactions_df = transactions_df.withColumn("timestamp", unix_timestamp(col("datetime")))
    window_spec = Window.partitionBy("cc_num").orderBy("timestamp")
    
    # Feature 1: Transaction velocity (count per card in last 1 hour)
    transactions_df = transactions_df.withColumn(
        "txn_velocity_1h",
        count("transaction_id").over(window_spec.rangeBetween(-3600, 0))
    )
    
    # Feature 2: Amount statistics (mean and stddev per card in last 24 hours)
    transactions_df = transactions_df.withColumn(
        "amount_mean_24h",
        mean("amount").over(window_spec.rangeBetween(-86400, 0))
    )
    transactions_df = transactions_df.withColumn(
        "amount_stddev_24h",
        stddev("amount").over(window_spec.rangeBetween(-86400, 0))
    )
    
    # Feature 3: Time since last transaction
    transactions_df = transactions_df.withColumn(
        "time_since_last_txn",
        col("timestamp") - lag("timestamp").over(window_spec)
    )
    
    # Feature 4: Amount deviation from mean
    transactions_df = transactions_df.withColumn(
        "amount_dev_from_mean",
        (col("amount") - col("amount_mean_24h")) / (col("amount_stddev_24h") + lit(1e-6))
    )
    
    # Feature 5: Geo distance from previous transaction
    transactions_df = transactions_df.withColumn(
        "prev_lat", lag("lat").over(window_spec)
    ).withColumn(
        "prev_long", lag("long").over(window_spec)
    )
    transactions_df = transactions_df.withColumn(
        "geo_distance",
        when(
            (col("prev_lat").isNotNull()) & (col("prev_long").isNotNull()),
            ((col("lat") - col("prev_lat"))**2 + (col("long") - col("prev_long"))**2)**0.5
        ).otherwise(lit(0.0))
    )
    
    # Drop intermediate columns
    transactions_df = transactions_df.drop("prev_lat", "prev_long", "timestamp")
    
    return transactions_df

# COMMAND ----------

@dlt.table(
    name=training_dataset_name,
    comment="Training dataset for fraud classification."
)
def create_training_dataset():
    # Read feature group
    feature_df = dlt.read(feature_group_name)
    
    # Assemble features
    feature_cols = [
        "txn_velocity_1h", "amount_mean_24h", "amount_stddev_24h",
        "time_since_last_txn", "amount_dev_from_mean", "geo_distance", "amount"
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    training_df = assembler.transform(feature_df)
    
    return training_df

# COMMAND ----------

@dlt.table(
    name=predictions_table_name,
    comment="Fraud probability predictions for scoring transactions."
)
def create_predictions_table():
    # Read feature group and score transactions
    feature_df = dlt.read(feature_group_name)
    score_df = spark.read.csv(score_path, header=True, inferSchema=True)
    
    # Feature Engineering for score_df
    score_df = score_df.withColumn("timestamp", unix_timestamp(col("datetime")))
    window_spec = Window.partitionBy("cc_num").orderBy("timestamp")
    
    # Reuse feature logic
    score_df = score_df.withColumn(
        "txn_velocity_1h",
        count("transaction_id").over(window_spec.rangeBetween(-3600, 0))
    )
    score_df = score_df.withColumn(
        "amount_mean_24h",
        mean("amount").over(window_spec.rangeBetween(-86400, 0))
    )
    score_df = score_df.withColumn(
        "amount_stddev_24h",
        stddev("amount").over(window_spec.rangeBetween(-86400, 0))
    )
    score_df = score_df.withColumn(
        "time_since_last_txn",
        col("timestamp") - lag("timestamp").over(window_spec)
    )
    score_df = score_df.withColumn(
        "amount_dev_from_mean",
        (col("amount") - col("amount_mean_24h")) / (col("amount_stddev_24h") + lit(1e-6))
    )
    score_df = score_df.withColumn(
        "prev_lat", lag("lat").over(window_spec)
    ).withColumn(
        "prev_long", lag("long").over(window_spec)
    )
    score_df = score_df.withColumn(
        "geo_distance",
        when(
            (col("prev_lat").isNotNull()) & (col("prev_long").isNotNull()),
            ((col("lat") - col("prev_lat"))**2 + (col("long") - col("prev_long"))**2)**0.5
        ).otherwise(lit(0.0))
    )
    
    # Drop intermediate columns
    score_df = score_df.drop("prev_lat", "prev_long", "timestamp")
    
    # Assemble features
    feature_cols = [
        "txn_velocity_1h", "amount_mean_24h", "amount_stddev_24h",
        "time_since_last_txn", "amount_dev_from_mean", "geo_distance", "amount"
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    score_df = assembler.transform(score_df)
    
    # Train model (using a subset of the feature group for training)
    training_df = dlt.read(training_dataset_name)
    (training_data, test_data) = training_df.randomSplit([0.8, 0.2], seed=42)
    rf = RandomForestClassifier(labelCol="is_fraud", featuresCol="features", numTrees=100)
    pipeline = Pipeline(stages=[rf])
    model = pipeline.fit(training_data)
    
    # Evaluate
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="is_fraud", metricName="areaUnderROC")
    roc_auc = evaluator.evaluate(predictions)
    print(f"ROC AUC: {{roc_auc}}")
    
    # Register Model
    model_path = f"/Users/{{os.getenv('USER')}}/{prefix}/fraud_model"
    model.write().overwrite().save(model_path)
    
    # Register in MLflow
    model_full_name = f"{catalog_name}.{schema_name_only}.{model_name}"
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    w.model_versions.create(name=model_full_name, source=model_path)
    
    # Score
    predictions_df = model.transform(score_df)
    predictions_df = predictions_df.select(
        "transaction_id",
        col("probability").getItem(1).alias("fraud_probability")
    )
    
    # Enable Low-Latency Lookup
    from databricks.sdk.service import ml
    w.online_tables.create(
        name=f"{catalog_name}.{schema_name_only}.{predictions_table_name}",
        spec=ml.OnlineTableSpec(
            primary_key_columns=["transaction_id"],
            source_table_full_name=f"{catalog_name}.{schema_name_only}.{predictions_table_name}",
        ),
    )
    
    return predictions_df