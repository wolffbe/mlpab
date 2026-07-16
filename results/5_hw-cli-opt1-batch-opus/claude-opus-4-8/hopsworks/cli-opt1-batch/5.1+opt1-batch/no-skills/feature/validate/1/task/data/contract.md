# Data contract — events export

Columns: `row_id` (string, unique record key), `account_id` (string), `event_time` (bigint, epoch milliseconds), `amount` (double), `category` (string).

A row is VALID only if ALL of the following hold:

1. **amount is present** — null/empty amounts are contract violations.
2. **amount is within [0, 10000]** (inclusive).
3. **category is one of**: `grocery`, `travel`, `salary`, `rent`, `other`.

Rows violating any rule must NOT be loaded; their ids must be reported.
