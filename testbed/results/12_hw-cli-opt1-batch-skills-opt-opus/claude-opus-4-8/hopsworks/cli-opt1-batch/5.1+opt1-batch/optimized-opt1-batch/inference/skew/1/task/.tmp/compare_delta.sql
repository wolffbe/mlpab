SELECT
  count(*) AS n_overlap,
  avg(abs(t.f1 - s.f1)) AS mad_f1,
  avg(abs(t.f2 - s.f2)) AS mad_f2,
  avg(abs(t.f3 - s.f3)) AS mad_f3,
  avg(abs(t.f4 - s.f4)) AS mad_f4,
  avg(abs(t.f5 - s.f5)) AS mad_f5,
  max(abs(t.f1 - s.f1)) AS max_f1,
  max(abs(t.f2 - s.f2)) AS max_f2,
  max(abs(t.f3 - s.f3)) AS max_f3,
  max(abs(t.f4 - s.f4)) AS max_f4,
  max(abs(t.f5 - s.f5)) AS max_f5
FROM delta.mlpab405514_featurestore.train_skew_1 t
JOIN delta.mlpab405514_featurestore.serve_skew_1 s
  ON t.entity_id = s.entity_id
