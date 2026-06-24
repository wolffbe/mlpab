"""Feature pipeline: build derivedc779d7 from rawac779d7 + rawbc779d7.

Runs on the Hopsworks cluster (PySpark). Inner-joins the two raw feature
groups on row_id, computes col_sum = round(a_val + b_val, 6), and writes a
derived feature group with online store enabled and provenance (parents)
registered so the derivation is traceable to its sources.
"""
import hopsworks
from pyspark.sql import functions as F

project = hopsworks.login()
fs = project.get_feature_store()

fga = fs.get_feature_group("rawac779d7", version=1)
fgb = fs.get_feature_group("rawbc779d7", version=1)

# Read each source as a Spark DataFrame, then inner-join in Spark on row_id
# (avoids hsfs auto-prefixing the right-side join columns).
dfa = fga.read().select(F.col("row_id"), F.col("a_val"))
dfb = fgb.read().select(F.col("row_id").alias("row_id_b"), F.col("b_val"))

# Inner join -> only row_ids present in BOTH sources.
joined = dfa.join(dfb, dfa["row_id"] == dfb["row_id_b"], "inner")
print("JOINED_COUNT", joined.count())

derived_df = joined.select(
    F.col("row_id").cast("string").alias("row_id"),
    F.round(F.col("a_val") + F.col("b_val"), 6).cast("double").alias("col_sum"),
)
print("DERIVED_COUNT", derived_df.count())
derived_df.show(5, truncate=False)

derived_fg = fs.get_or_create_feature_group(
    name="derivedc779d7",
    version=1,
    primary_key=["row_id"],
    online_enabled=True,
    parents=[fga, fgb],
    description="row_id, col_sum=round(a_val+b_val,6); inner join of rawac779d7 & rawbc779d7",
)
derived_fg.insert(derived_df)
print("INSERT_DONE", derived_fg.name, "id=", derived_fg.id)
