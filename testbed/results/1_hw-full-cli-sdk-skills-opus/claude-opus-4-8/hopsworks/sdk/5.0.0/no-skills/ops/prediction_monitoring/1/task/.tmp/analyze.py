import hopsworks
proj = hopsworks.login()
fs = proj.get_feature_store()
fg = fs.get_feature_group('prediction_log', version=1)

# Find offline table name
tbl = fg._get_table_name() if hasattr(fg, '_get_table_name') else None
print('table:', tbl)

# Daily aggregation via platform query engine (offline store SQL)
q = f"""
SELECT date(ts) AS d,
       count(*) AS n,
       avg(prediction) AS mean_pred,
       stddev(prediction) AS std_pred,
       min(prediction) AS min_pred,
       max(prediction) AS max_pred
FROM `{tbl}`
GROUP BY date(ts)
ORDER BY d
"""
res = fs.sql(q, online=False)
print(type(res))
print(res.to_string())
res.to_csv('.tmp/daily.csv', index=False)
