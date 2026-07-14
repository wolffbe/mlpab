CREATE OR REPLACE MODEL `mlpab_mlpab08f695.ccmodel76ccb2`
OPTIONS(
  model_type = 'BOOSTED_TREE_CLASSIFIER',
  input_label_cols = ['is_fraud'],
  auto_class_weights = TRUE,
  data_split_method = 'AUTO_SPLIT',
  max_iterations = 50,
  learn_rate = 0.1,
  early_stop = TRUE,
  model_registry = 'vertex_ai',
  vertex_ai_model_id = 'mlpab08f695_ccmodel76ccb2',
  vertex_ai_model_version_aliases = ['default']
) AS
SELECT
  is_fraud, amount, category, log_amt, hour, dow, is_night,
  geo_dist_km, amt_over_avg, amt_z, card_txn_cnt,
  velocity_1h, velocity_24h, secs_since_prev
FROM `mlpab_mlpab08f695.cctd76ccb2`;
