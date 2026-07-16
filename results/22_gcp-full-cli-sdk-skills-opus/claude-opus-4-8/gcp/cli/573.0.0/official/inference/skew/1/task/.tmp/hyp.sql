SELECT
  MAX(ABS(t.f2 - LN(1 + s.f2)))            AS max_err_log1p_serve,
  MAX(ABS(s.f2 - (EXP(t.f2) - 1)))         AS max_err_expm1_train,
  MAX(ABS(t.f2 - LN(s.f2)))                AS max_err_ln_serve
FROM mlpab_mlpab5a918e.training_sample t
JOIN mlpab_mlpab5a918e.serving_log s USING(entity_id)
