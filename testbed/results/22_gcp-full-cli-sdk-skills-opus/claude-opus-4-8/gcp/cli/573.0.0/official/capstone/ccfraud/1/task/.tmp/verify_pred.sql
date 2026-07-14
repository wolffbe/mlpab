SELECT COUNT(*) AS n, COUNT(DISTINCT transaction_id) AS n_ids,
       MIN(fraud_probability) AS mn, MAX(fraud_probability) AS mx,
       AVG(fraud_probability) AS avg_p,
       SUM(IF(fraud_probability IS NULL,1,0)) AS n_null,
       SUM(IF(fraud_probability BETWEEN 0 AND 1,0,1)) AS out_of_range
FROM `mlpab_mlpab08f695.ccpred76ccb2`;
