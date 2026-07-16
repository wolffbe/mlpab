WITH stats AS (
  SELECT
    avg(f1) AS m1, stddev_pop(f1) AS s1,
    avg(f2) AS m2, stddev_pop(f2) AS s2,
    avg(f3) AS m3, stddev_pop(f3) AS s3,
    avg(f4) AS m4, stddev_pop(f4) AS s4
  FROM raw_train_1
),
combined AS (
  SELECT row_id, 'train' AS split, f1, f2, f3, f4 FROM raw_train_1
  UNION ALL
  SELECT row_id, 'serve' AS split, f1, f2, f3, f4 FROM raw_serve_1
)
SELECT
  c.row_id AS row_id,
  c.split AS split,
  ROUND((c.f1 - s.m1) / s.s1, 6) AS f1,
  ROUND((c.f2 - s.m2) / s.s2, 6) AS f2,
  ROUND((c.f3 - s.m3) / s.s3, 6) AS f3,
  ROUND((c.f4 - s.m4) / s.s4, 6) AS f4
FROM combined c CROSS JOIN stats s
ORDER BY c.row_id
