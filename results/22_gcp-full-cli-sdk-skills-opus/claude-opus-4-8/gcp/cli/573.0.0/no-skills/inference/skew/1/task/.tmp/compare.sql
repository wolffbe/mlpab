SELECT
  COUNT(*) AS n,
  ROUND(AVG(ABS(t.f1 - s.f1)),4) AS mad_f1,
  ROUND(AVG(ABS(t.f2 - s.f2)),4) AS mad_f2,
  ROUND(AVG(ABS(t.f3 - s.f3)),4) AS mad_f3,
  ROUND(AVG(ABS(t.f4 - s.f4)),4) AS mad_f4,
  ROUND(AVG(ABS(t.f5 - s.f5)),4) AS mad_f5,
  ROUND(CORR(t.f1,s.f1),4) AS corr_f1,
  ROUND(CORR(t.f2,s.f2),4) AS corr_f2,
  ROUND(CORR(t.f3,s.f3),4) AS corr_f3,
  ROUND(CORR(t.f4,s.f4),4) AS corr_f4,
  ROUND(CORR(t.f5,s.f5),4) AS corr_f5,
  ROUND(AVG(s.f1 - t.f1),4) AS bias_f1,
  ROUND(AVG(s.f2 - t.f2),4) AS bias_f2,
  ROUND(AVG(s.f3 - t.f3),4) AS bias_f3,
  ROUND(AVG(s.f4 - t.f4),4) AS bias_f4,
  ROUND(AVG(s.f5 - t.f5),4) AS bias_f5
FROM `***REDACTED***.mlpab_mlpab75cdd3.training_sample` t
JOIN `***REDACTED***.mlpab_mlpab75cdd3.serving_log` s
USING (entity_id)
