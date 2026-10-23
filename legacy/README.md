# Historical experiment scripts

These scripts are retained byte-for-byte as historical research evidence.
They were moved from the repository root to simplify the supported project
layout. Their original root-level outputs remain in place; existing result
tables and plots were not moved or regenerated.

Use `python -m calibrationfreebci` for the tested pipeline. These archived scripts
are not supported production entry points: several contain superseded timing,
assumptions or configuration, and some can initiate downloads. Scripts deriving
paths from `__file__` still reflect their original root-level layout and should
not be rerun from this folder as if they were updated experiments.

To inspect or reproduce the original layout, use the preserved ZIP documented in
[PRESERVATION.md](../PRESERVATION.md) or the Git commit before this move. Do not
overwrite historical outputs when reproducing an old experiment. The corrected
`04_motor_imagery_epochs.py` visualization and current `main.py` entry point
remain at the repository root because they use the supported package.
