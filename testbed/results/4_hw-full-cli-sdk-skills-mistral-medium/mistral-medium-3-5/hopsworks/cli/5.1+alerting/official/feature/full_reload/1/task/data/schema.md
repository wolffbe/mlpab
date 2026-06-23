# Schema

## initial_export.csv (the original schema)
- **row_id** (string): unique record key
- **name** (string)
- **balance_eur** (double): balance, always in EUR
- **updated_at** (bigint): epoch MILLISECONDS, the event-time column

## reload/new_export.csv (the NEW, breaking schema)
A complete re-export from the upstream source. Columns were renamed and extended; every row was re-issued with new values; some old row_ids no longer exist and new ones were added.
- **row_id** (string): unique record key
- **full_name** (string): replaces `name`
- **balance** (double): replaces `balance_eur`; currency now varies
- **currency** (string): ISO code
- **updated_at** (bigint): epoch MILLISECONDS, the event-time column
