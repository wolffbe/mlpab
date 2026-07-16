SELECT
  COUNT(*) AS n,
  AVG(ABS(t.f1-s.f1)) AS mad_f1,
  AVG(ABS(t.f2-s.f2)) AS mad_f2,
  AVG(ABS(t.f3-s.f3)) AS mad_f3,
  AVG(ABS(t.f4-s.f4)) AS mad_f4,
  AVG(ABS(t.f5-s.f5)) AS mad_f5,
  CORR(t.f1,s.f1) AS corr_f1,
  CORR(t.f2,s.f2) AS corr_f2,
  CORR(t.f3,s.f3) AS corr_f3,
  CORR(t.f4,s.f4) AS corr_f4,
  CORR(t.f5,s.f5) AS corr_f5
FROM mlpab_mlpab5a918e.training_sample t
JOIN mlpab_mlpab5a918e.serving_log s USING(entity_id)
