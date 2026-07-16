import dlt

@dlt.table(
    name="mlpabbc4768_prediction_log",
    comment="Prediction log for model monitoring"
)
def create_prediction_log():
    return spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/Volumes/workspace/mlpabbc4768/mlpabbc4768_volume/prediction_log.csv")