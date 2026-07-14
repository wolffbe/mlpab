DROP TABLE IF EXISTS `mlpab_mlpab08f695.ccpred76ccb2`;
CREATE TABLE `mlpab_mlpab08f695.ccpred76ccb2`
CLUSTER BY transaction_id AS
SELECT
  transaction_id,
  (SELECT p.prob FROM UNNEST(predicted_is_fraud_probs) AS p WHERE p.label = 1) AS fraud_probability,
  CURRENT_TIMESTAMP() AS feature_timestamp
FROM ML.PREDICT(
  MODEL `mlpab_mlpab08f695.ccmodel76ccb2`,
  (SELECT * FROM `mlpab_mlpab08f695.score_features`)
);
