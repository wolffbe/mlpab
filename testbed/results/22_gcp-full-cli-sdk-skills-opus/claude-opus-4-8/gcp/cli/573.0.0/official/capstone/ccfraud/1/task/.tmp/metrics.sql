CREATE OR REPLACE TABLE `mlpab_mlpab08f695.ccmodel76ccb2_metrics` AS
SELECT CURRENT_TIMESTAMP() AS evaluated_at, * FROM ML.EVALUATE(MODEL `mlpab_mlpab08f695.ccmodel76ccb2`);
