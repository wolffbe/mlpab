WITH stats AS (
  SELECT
    AVG(f1) AS m1, AVG(f2) AS m2, AVG(f3) AS m3, AVG(f4) AS m4,
    STDDEV_POP(f1) AS s1, STDDEV_POP(f2) AS s2, STDDEV_POP(f3) AS s3, STDDEV_POP(f4) AS s4
  FROM DSET._stg_train
),
combined AS (
  SELECT row_id, 'train' AS split, f1, f2, f3, f4 FROM DSET._stg_train
  UNION ALL
  SELECT row_id, 'serve' AS split, f1, f2, f3, f4 FROM DSET._stg_serve
)
SELECT
  c.row_id,
  c.split,
  ROUND((c.f1 - s.m1)/s.s1, 6) AS f1,
  ROUND((c.f2 - s.m2)/s.s2, 6) AS f2,
  ROUND((c.f3 - s.m3)/s.s3, 6) AS f3,
  ROUND((c.f4 - s.m4)/s.s4, 6) AS f4
FROM combined c CROSS JOIN stats s
