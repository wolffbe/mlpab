import hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
schema = fs.offline_featurestore_name  # <project>_featurestore
print("offline schema:", schema)

trino_api = project.get_trino_api()
conn = trino_api.connect(catalog="delta", schema=schema, verify=False)
cur = conn.cursor()

tbl = f"delta.{schema}.scaled1f3dc5_stg_1"
cur.execute(
    "SELECT count(*) n, count_if(split='train') ntrain, count_if(split='serve') nserve, "
    f"avg(CASE WHEN split='train' THEN f1 END), stddev_pop(CASE WHEN split='train' THEN f1 END) "
    f"FROM {tbl}"
)
print("counts+stat:", cur.fetchall())
