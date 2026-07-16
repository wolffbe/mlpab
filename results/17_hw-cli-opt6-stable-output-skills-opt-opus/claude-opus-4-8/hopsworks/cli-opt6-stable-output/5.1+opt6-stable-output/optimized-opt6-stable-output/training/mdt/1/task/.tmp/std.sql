WITH stats AS (
  SELECT avg(f1) m1, avg(f2) m2, avg(f3) m3, avg(f4) m4,
         stddev_pop(f1) s1, stddev_pop(f2) s2, stddev_pop(f3) s3, stddev_pop(f4) s4
  FROM raw_train_sd8_1
)
SELECT t.row_id AS row_id, 'train' AS split,
  round((t.f1-s.m1)/s.s1, 6) AS f1,
  round((t.f2-s.m2)/s.s2, 6) AS f2,
  round((t.f3-s.m3)/s.s3, 6) AS f3,
  round((t.f4-s.m4)/s.s4, 6) AS f4
FROM raw_train_sd8_1 t CROSS JOIN stats s
UNION ALL
SELECT v.row_id AS row_id, 'serve' AS split,
  round((v.f1-s.m1)/s.s1, 6) AS f1,
  round((v.f2-s.m2)/s.s2, 6) AS f2,
  round((v.f3-s.m3)/s.s3, 6) AS f3,
  round((v.f4-s.m4)/s.s4, 6) AS f4
FROM raw_serve_sd8_1 v CROSS JOIN stats s
