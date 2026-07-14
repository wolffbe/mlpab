SELECT
  entity_id,
  ROUND(t.f2,4) AS train_f2,
  ROUND(s.f2,4) AS serve_f2,
  ROUND(s.f2 - t.f2,4) AS diff,
  ROUND(SAFE_DIVIDE(s.f2, t.f2),4) AS ratio
FROM `***REDACTED***.mlpab_mlpab75cdd3.training_sample` t
JOIN `***REDACTED***.mlpab_mlpab75cdd3.serving_log` s
USING (entity_id)
ORDER BY t.f2
LIMIT 12
