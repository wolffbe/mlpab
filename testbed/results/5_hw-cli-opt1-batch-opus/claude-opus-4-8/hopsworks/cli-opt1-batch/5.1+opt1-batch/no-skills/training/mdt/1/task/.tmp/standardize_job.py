import hopsworks
from pyspark.sql import functions as F

project = hopsworks.login()
fs = project.get_feature_store()

train_fg = fs.get_feature_group("scaled_raw_train", version=1)
serve_fg = fs.get_feature_group("scaled_raw_serve", version=1)

train_df = train_fg.read()
serve_df = serve_fg.read()

feats = ["f1", "f2", "f3", "f4"]

# Train-only statistics: mean and POPULATION std (no Bessel correction)
agg_exprs = []
for f in feats:
    agg_exprs.append(F.mean(F.col(f)).alias("m_" + f))
    agg_exprs.append(F.stddev_pop(F.col(f)).alias("s_" + f))
stats = train_df.agg(*agg_exprs).collect()[0]

means = {f: float(stats["m_" + f]) for f in feats}
stds = {f: float(stats["s_" + f]) for f in feats}
print("means=", means)
print("stds=", stds)


def standardize(df, split_label):
    cols = [F.col("row_id").alias("row_id"), F.lit(split_label).alias("split")]
    for f in feats:
        cols.append(
            F.round((F.col(f) - F.lit(means[f])) / F.lit(stds[f]), 6).alias(f)
        )
    return df.select(*cols)


train_std = standardize(train_df, "train")
serve_std = standardize(serve_df, "serve")
result = train_std.unionByName(serve_std)

result = result.select("row_id", "split", "f1", "f2", "f3", "f4")
print("result count=", result.count())
result.show(5, False)

out_fg = fs.get_or_create_feature_group(
    name="scaledd69ec7",
    version=1,
    primary_key=["row_id"],
    online_enabled=True,
    description="Standardized features (train+serve) using train-only mean and population std.",
)

out_fg.insert(result)
print("DONE: inserted", result.count(), "rows into scaledd69ec7 v1")
