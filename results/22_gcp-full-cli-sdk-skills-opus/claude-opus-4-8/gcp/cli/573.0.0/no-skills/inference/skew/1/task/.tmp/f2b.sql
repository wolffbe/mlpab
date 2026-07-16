SELECT
  ROUND(MAX(ABS(s.f2 - (EXP(t.f2) - 1))),6) AS max_err_expm1,
  ROUND(MAX(ABS(t.f2 - LN(1 + s.f2))),6) AS max_err_log1p
FROM `***REDACTED***.mlpab_mlpab75cdd3.training_sample` t
JOIN `***REDACTED***.mlpab_mlpab75cdd3.serving_log` s
USING (entity_id)
