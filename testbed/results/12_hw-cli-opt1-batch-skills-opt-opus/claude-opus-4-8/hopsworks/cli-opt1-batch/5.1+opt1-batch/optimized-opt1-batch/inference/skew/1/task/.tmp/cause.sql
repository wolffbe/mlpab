SELECT t.entity_id, t.f5 AS train_f5, s.f5 AS serve_f5,
       s.f5 / t.f5 AS ratio,
       s.f5 - t.f5 AS diff,
       t.f1, t.f2, t.f3, t.f4
FROM delta.mlpab405514_featurestore.train_skew_1 t
JOIN delta.mlpab405514_featurestore.serve_skew_1 s ON t.entity_id = s.entity_id
ORDER BY t.entity_id
LIMIT 12
