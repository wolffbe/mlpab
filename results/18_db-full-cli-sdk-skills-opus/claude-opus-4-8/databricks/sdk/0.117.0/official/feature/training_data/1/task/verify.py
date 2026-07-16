from db import *
T = f'{CAT}.{SCH}'

# 1. Recompute expected most-recent transaction (incl. late) per label and compare
q = f'''
WITH expected AS (
  SELECT l.account_id, l.label_time,
    max_by(t.amount, t.event_time) amt, max_by(t.balance, t.event_time) bal
  FROM {T}.src_labels l JOIN {T}.src_transactions t
    ON t.account_id=l.account_id AND t.event_time<=l.label_time
  GROUP BY l.account_id, l.label_time
)
SELECT count(*) FROM {T}.churntraining605fb7 c JOIN expected e
  ON c.account_id=e.account_id AND c.label_time=e.label_time
WHERE c.amount<>e.amt OR c.balance<>e.bal'''
print('tx mismatches:', rows(sql(q)))

# 2. Same independent recompute for the other three tables
for tbl, col in [('src_profiles', 'credit_score'), ('src_activity', 'sessions_7d'), ('src_health', 'health_score')]:
    q = f'''
    WITH expected AS (
      SELECT l.account_id, l.label_time, max_by(s.{col}, s.event_time) v
      FROM {T}.src_labels l JOIN {T}.{tbl} s
        ON s.account_id=l.account_id AND s.event_time<=l.label_time
      GROUP BY l.account_id, l.label_time
    )
    SELECT count(*) FROM {T}.churntraining605fb7 c JOIN expected e
      ON c.account_id=e.account_id AND c.label_time=e.label_time
    WHERE c.{col}<>e.v'''
    print(f'{col} mismatches:', rows(sql(q)))

# 3. profiles tier consistency with chosen credit_score row (most-recent profile)
q = f'''
WITH expected AS (
  SELECT l.account_id, l.label_time, max_by(p.tier, p.event_time) v
  FROM {T}.src_labels l JOIN {T}.src_profiles p
    ON p.account_id=l.account_id AND p.event_time<=l.label_time
  GROUP BY l.account_id, l.label_time
)
SELECT count(*) FROM {T}.churntraining605fb7 c JOIN expected e
  ON c.account_id=e.account_id AND c.label_time=e.label_time
WHERE c.tier<>e.v'''
print('tier mismatches:', rows(sql(q)))
