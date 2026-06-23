WITH stats AS (
  SELECT avg(f1) m1, avg(f2) m2, avg(f3) m3, avg(f4) m4,
         stddev_pop(f1) s1, stddev_pop(f2) s2, stddev_pop(f3) s3, stddev_pop(f4) s4
  FROM scaled3effa1_train_raw_1
),
both AS (
  SELECT row_id, 'train' AS split, f1, f2, f3, f4 FROM scaled3effa1_train_raw_1
  UNION ALL
  SELECT row_id, 'serve' AS split, f1, f2, f3, f4 FROM scaled3effa1_serve_raw_1
)
SELECT b.row_id, b.split,
  round((b.f1 - s.m1)/s.s1, 6) AS f1,
  round((b.f2 - s.m2)/s.s2, 6) AS f2,
  round((b.f3 - s.m3)/s.s3, 6) AS f3,
  round((b.f4 - s.m4)/s.s4, 6) AS f4
FROM both b CROSS JOIN stats s
