-- Compute correlation between each feature and the label
SELECT
  corr(f1, label) AS f1_corr,
  corr(f2, label) AS f2_corr,
  corr(f3, label) AS f3_corr,
  corr(f4, label) AS f4_corr,
  corr(f5, label) AS f5_corr,
  corr(f6, label) AS f6_corr
FROM workspace.mlpabb1ccad.training_data;