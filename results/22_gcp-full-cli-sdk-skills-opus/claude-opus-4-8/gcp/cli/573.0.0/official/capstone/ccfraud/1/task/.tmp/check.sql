SELECT 'train' AS t, COUNT(*) AS n, SUM(is_fraud) AS fraud, MIN(datetime) AS mn, MAX(datetime) AS mx FROM `mlpab_mlpab08f695.raw_transactions`
UNION ALL
SELECT 'score', COUNT(*), NULL, MIN(datetime), MAX(datetime) FROM `mlpab_mlpab08f695.raw_score`
