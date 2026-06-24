SELECT substr(event_time,1,10) AS d,
       round(avg(f1),4) m1, round(stddev(f1),4) s1,
       round(avg(f2),4) m2, round(stddev(f2),4) s2,
       round(avg(f3),4) m3, round(stddev(f3),4) s3,
       round(avg(f4),4) m4, round(stddev(f4),4) s4,
       round(avg(f5),4) m5, round(stddev(f5),4) s5,
       round(avg(f6),4) m6, round(stddev(f6),4) s6
FROM delta.mlpabd1cc09_featurestore.drift_features_1
GROUP BY substr(event_time,1,10)
ORDER BY d
