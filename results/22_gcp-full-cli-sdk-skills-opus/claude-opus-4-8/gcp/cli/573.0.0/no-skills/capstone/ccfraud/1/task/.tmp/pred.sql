CREATE OR REPLACE TABLE `***REDACTED***.mlpab_mlpab5d6311.ccpred76ccb2` AS
SELECT
  transaction_id,
  (SELECT p.prob FROM UNNEST(predicted_is_fraud_probs) AS p WHERE p.label = 1) AS fraud_probability
FROM ML.PREDICT(
  MODEL `***REDACTED***.mlpab_mlpab5d6311.ccmodel76ccb2`,
  (SELECT * FROM `***REDACTED***.mlpab_mlpab5d6311.score_features`)
)
