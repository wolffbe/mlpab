CREATE OR REPLACE MODEL `***REDACTED***.mlpab_mlpab5d6311.ccmodel76ccb2`
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['is_fraud'],
  auto_class_weights=TRUE,
  data_split_method='RANDOM',
  data_split_eval_fraction=0.2,
  max_iterations=50,
  model_registry='VERTEX_AI',
  vertex_ai_model_id='mlpab5d6311_ccmodel76ccb2'
) AS
SELECT category, amount, hour, dow, log_amount, amount_vs_avg, amount_z, geo_dist,
       velocity_1h, velocity_24h, secs_since_prev, is_fraud
FROM `***REDACTED***.mlpab_mlpab5d6311.cctd76ccb2`
