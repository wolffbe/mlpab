import databricks.sdk

w = databricks.sdk.WorkspaceClient()
SCHEMA = "workspace.mlpab98b0c2"
WH = "4dfab06c923fe3cc"


def run(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout="50s")
    return r.result.data_array


# relationships for f3: try s = a*t, s = t+b, s = log(t), s = t^2, corr of s vs ln t etc.
rows = run(f"""
SELECT
  corr(s.f3, t.f3) c_lin,
  corr(s.f3, ln(t.f3)) c_serve_vs_lnT,
  corr(ln(s.f3), t.f3) c_lnServe_vs_T,
  avg(s.f3/t.f3) ratio,
  avg(s.f3 - t.f3) diff,
  avg(s.f3 - ln(t.f3)) diff_lnT,
  avg(s.f3) avgS, avg(t.f3) avgT
FROM {SCHEMA}.train t JOIN {SCHEMA}.serve s ON t.entity_id=s.entity_id
""")
print("c_lin, c_serve_vs_lnT, c_lnServe_vs_T, avg(s/t), avg(s-t), avg(s-lnT), avgS, avgT")
print([round(float(x),5) for x in rows[0]])

# sample side by side
rows = run(f"""
SELECT t.entity_id, t.f3 train_f3, s.f3 serve_f3, ln(t.f3) ln_train, exp(s.f3) exp_serve
FROM {SCHEMA}.train t JOIN {SCHEMA}.serve s ON t.entity_id=s.entity_id
ORDER BY t.entity_id LIMIT 12
""")
print("\nentity train_f3 serve_f3 ln(train) exp(serve)")
for r in rows:
    print([str(r[0])] + [round(float(x),4) for x in r[1:]])
