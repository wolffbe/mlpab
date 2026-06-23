WITH stats AS (
  SELECT avg(f1) m1, stddev_pop(f1) s1, avg(f2) m2, stddev_pop(f2) s2,
         avg(f3) m3, stddev_pop(f3) s3, avg(f4) m4, stddev_pop(f4) s4
  FROM delta.mlpab19e298_featurestore.stg_train63801_1
)
SELECT row_id, 'train' AS split,
  round((f1-m1)/s1,6) AS f1, round((f2-m2)/s2,6) AS f2,
  round((f3-m3)/s3,6) AS f3, round((f4-m4)/s4,6) AS f4
FROM delta.mlpab19e298_featurestore.stg_train63801_1, stats
UNION ALL
SELECT row_id, 'serve' AS split,
  round((f1-m1)/s1,6) AS f1, round((f2-m2)/s2,6) AS f2,
  round((f3-m3)/s3,6) AS f3, round((f4-m4)/s4,6) AS f4
FROM delta.mlpab19e298_featurestore.stg_serve63801_1, stats
