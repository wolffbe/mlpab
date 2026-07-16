# Schema

Three batch files of one `accounts` table, delivered OUT OF ORDER: the same `row_id` may appear in more than one batch, and the row with the LATEST `updated_at` is the correct, current state (later rows are corrections of earlier ones).

- **row_id** (string): unique record key
- **status** (string): active | dormant | closed
- **balance** (double)
- **updated_at** (bigint): revision time as epoch MILLISECONDS — register it as the event-time column
