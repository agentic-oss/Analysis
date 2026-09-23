# Data Schema & Standards

## Raw & Processed Observation Schema
- `timestamp`: String (`YYYY-MM-DD`)
- `open`: Float64
- `high`: Float64
- `low`: Float64
- `close`: Float64
- `volume`: Float64
- `provider`: String
- `source`: String
- `retrieval_timestamp`: ISO 8601 String
- `instrument`: String
- `timezone`: String

## Quality Classifications
Observations are classified into:
- `valid`: All checks passed.
- `suspicious`: Duplicate timestamps or abnormal price jumps (>20%).
- `invalid`: Non-positive prices or High < Low / Open / Close logic violations.
