# Schema

Both export files share one schema; together they contain the full table, but their row ranges OVERLAP (the second file is a re-delivery that includes the tail of the first). `row_id` uniquely identifies a row.

- **row_id** (string): unique record key
- **account_id** (string)
- **event_time** (bigint): when the row became valid, as epoch MILLISECONDS — register it as the event-time column
- **amount** (double)
- **category** (string)
