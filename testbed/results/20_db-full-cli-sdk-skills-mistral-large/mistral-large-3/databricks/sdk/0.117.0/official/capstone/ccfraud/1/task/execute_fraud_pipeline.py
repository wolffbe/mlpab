#!/usr/bin/env python3
"""
Executes the fraud FTI pipeline directly using the Databricks SDK.
"""
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, lag, stddev, mean, count, lit, when
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# Initialize Spark Session (for local testing only, not used in job)
spark = SparkSession.builder.getOrCreate()

# Config
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.mlpab21b96f
catalog_name = schema_name.split(".")[0]  # workspace
schema_name_only = schema_name.split(".")[1]  # mlpab21b96f
prefix = os.getenv("MLPAB_DATABRICKS_PREFIX")  # mlpab21b96f
job_name = f"{prefix}_fraud_pipeline"

# Feature Group, Dataset, Model, and Predictions Table Names
feature_group_name = "cctxne0b071"
training_dataset_name = "cctde0b071"
model_name = "ccmodele0b071"
predictions_table_name = "ccprede0b071"

# Paths
volume_path = f"/Volumes/{catalog_name}/{schema_name_only}/{prefix}_fraud_data"
transactions_path = f"{volume_path}/transactions.csv"
score_path = f"{volume_path}/score_transactions.csv"

# Define the pipeline function
def run_fraud_pipeline():
    from pyspark.sql.functions import col, unix_timestamp, lag, stddev, mean, count, lit, when
    from pyspark.sql.window import Window
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml import Pipeline
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service import ml

    # Read data
    print("Reading transactions data...")
    transactions_df = spark.read.csv(transactions_path, header=True, inferSchema=True)
    score_df = spark.read.csv(score_path, header=True, inferSchema=True)

    # Feature Engineering
    print("Engineering fraud features...")
    # Convert datetime to timestamp
    transactions_df = transactions_df.withColumn("timestamp", unix_timestamp(col("datetime")))

    # Window for time-based features
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

    # Write to Feature Group
    feature_group_full_name = f"{catalog_name}.{schema_name_only}.{feature_group_name}"
    print(f"Writing feature group: {feature_group_full_name}")
    transactions_df.write.format("delta").mode("overwrite").saveAsTable(feature_group_full_name)

    # Assemble Training Dataset
    print("Assembling training dataset...")
    feature_cols = [
        "txn_velocity_1h", "amount_mean_24h", "amount_stddev_24h",
        "time_since_last_txn", "amount_dev_from_mean", "geo_distance", "amount"
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    training_df = assembler.transform(transactions_df)

    # Split into train/test
    (training_data, test_data) = training_df.randomSplit([0.8, 0.2], seed=42)

    # Train Classifier
    print("Training classifier...")
    rf = RandomForestClassifier(labelCol="is_fraud", featuresCol="features", numTrees=100)
    pipeline = Pipeline(stages=[rf])
    model = pipeline.fit(training_data)

    # Evaluate
    print("Evaluating model...")
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="is_fraud", metricName="areaUnderROC")
    roc_auc = evaluator.evaluate(predictions)
    print(f"ROC AUC: {roc_auc}")

    # Register Model
    print("Registering model...")
    model_path = f"/Users/{os.getenv('USER')}/{prefix}/fraud_model"
    model.write().overwrite().save(model_path)

    # Register in MLflow
    model_full_name = f"{catalog_name}.{schema_name_only}.{model_name}"
    w = WorkspaceClient()
    w.model_versions.create(name=model_full_name, source=model_path)

    # Score score_transactions.csv
    print("Scoring transactions...")
    # Apply the same feature engineering to score_df
    score_df = score_df.withColumn("timestamp", unix_timestamp(col("datetime")))

    # Reuse the same window and feature logic
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
    ^
SyntaxError: invalid syntax

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
      File "/Users/wolffbe/workspace/banter/testbed/results/20_db-full-cli-sdk-skills-mistral-large/mistral-large-3/databricks/sdk/0.117.0/official/capstone/ccfraud/1/task/execute_fraud_pipeline.py", line 180, in <module>
        run_fraud_pipeline()
      File "/Users/wolffbe/workspace/banter/testbed/results/20_db-full-cli-sdk-skills-mistral-large/mistral-large-3/databricks/sdk/0.117.0/official/capstone/ccfraud/1/task/execute_fraud_pipeline.py", line 150, in run_fraud_pipeline
        score_df = score_df.withColumn(
    ValueError: Mixing Spark transformations with local execution is not supported. Run the pipeline on the platform.