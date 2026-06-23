WITH t AS (
  SELECT * FROM iceberg.mlpab9f55c7_featurestore.scaled_raw_train_1
),
s AS (
  SELECT * FROM iceberg.mlpab9f55c7_featurestore.scaled_raw_serve_1
),
stats AS (
  SELECT avg(f1) AS m1, stddev_pop(f1) AS d1,
         avg(f2) AS m2, stddev_pop(f2) AS d2,
         avg(f3) AS m3, stddev_pop(f3) AS d3,
         avg(f4) AS m4, stddev_pop(f4) AS d4
  FROM t
)
SELECT t.row_id AS row_id, 'train' AS split,
  round((t.f1 - stats.m1) / stats.d1, 6) AS f1,
  round((t.f2 - stats.m2) / stats.d2, 6) AS f2,
  round((t.f3 - stats.m3) / stats.d3, 6) AS f3,
  round((t.f4 - stats.m4) / stats.d4, 6) AS f4
FROM t CROSS JOIN stats
UNION ALL
SELECT s.row_id AS row_id, 'serve' AS split,
  round((s.f1 - stats.m1) / stats.d1, 6) AS f1,
  round((s.f2 - stats.m2) / stats.d2, 6) AS f2,
  round((s.f3 - stats.m3) / stats.d3, 6) AS f3,
  round((s.f4 - stats.m4) / stats.d4, 6) AS f4
FROM s CROSS JOIN stats
