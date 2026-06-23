SELECT split, count(*) AS n,
  round(avg(f1),4) AS m1, round(stddev_pop(f1),4) AS s1,
  round(avg(f4),4) AS m4, round(stddev_pop(f4),4) AS s4
FROM scaled3effa1_1
GROUP BY split
ORDER BY split
