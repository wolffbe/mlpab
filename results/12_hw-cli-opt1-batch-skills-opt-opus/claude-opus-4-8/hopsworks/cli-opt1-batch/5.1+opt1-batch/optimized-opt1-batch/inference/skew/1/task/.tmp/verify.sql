SELECT
  count(*) AS n,
  max(abs(s.f5 - (exp(t.f5) - 1))) AS max_err_expm1,
  max(abs(t.f5 - ln(1 + s.f5))) AS max_err_log1p
FROM delta.mlpab405514_featurestore.train_skew_1 t
JOIN delta.mlpab405514_featurestore.serve_skew_1 s ON t.entity_id = s.entity_id
