# Schema

Daily increment files of one events table; each file holds one day's new rows (no overlaps between files). `row_id` uniquely identifies a row.

- **row_id** (string): unique record key
- **account_id** (string)
- **event_time** (bigint): when the row became valid, as epoch MILLISECONDS — register it as the event-time column
- **amount** (double)
- **category** (string)
