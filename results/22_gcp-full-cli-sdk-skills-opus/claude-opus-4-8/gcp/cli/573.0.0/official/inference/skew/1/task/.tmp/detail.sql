SELECT
  t.entity_id,
  t.f2 AS train_f2,
  s.f2 AS serve_f2,
  SAFE_DIVIDE(s.f2, t.f2) AS ratio,
  s.f2 - t.f2 AS diff
FROM mlpab_mlpab5a918e.training_sample t
JOIN mlpab_mlpab5a918e.serving_log s USING(entity_id)
ORDER BY t.entity_id
LIMIT 12
