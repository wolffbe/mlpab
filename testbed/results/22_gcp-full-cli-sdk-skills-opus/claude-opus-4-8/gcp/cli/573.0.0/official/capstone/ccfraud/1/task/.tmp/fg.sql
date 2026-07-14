CREATE OR REPLACE TABLE `mlpab_mlpab08f695.cctxn76ccb2` AS
SELECT
  transaction_id, cc_num, datetime, feature_timestamp, is_fraud,
  amount, category, log_amt, hour, dow, is_night,
  geo_dist_km, amt_over_avg, amt_z, card_txn_cnt,
  velocity_1h, velocity_24h, secs_since_prev
FROM `mlpab_mlpab08f695.feat_all`
WHERE src = 'train';

CREATE OR REPLACE TABLE `mlpab_mlpab08f695.score_features` AS
SELECT
  transaction_id, cc_num, datetime, feature_timestamp,
  amount, category, log_amt, hour, dow, is_night,
  geo_dist_km, amt_over_avg, amt_z, card_txn_cnt,
  velocity_1h, velocity_24h, secs_since_prev
FROM `mlpab_mlpab08f695.feat_all`
WHERE src = 'score';
