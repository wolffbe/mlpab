SELECT substr(ts,1,10) AS day, count(*) AS n, round(avg(prediction),4) AS mean, round(stddev(prediction),4) AS sd FROM prediction_log_1 GROUP BY substr(ts,1,10) ORDER BY day
