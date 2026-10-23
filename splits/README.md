# Split-manifest availability

The validated local manifest is `outputs/v4_physionet2019/splits/setA_patient_split.csv` (20,336 assignments; 406,133 bytes). It contains PhysioNet patient file identifiers and is intentionally not copied into this public release candidate pending data-use/privacy review.

Authorized researchers should regenerate it deterministically with seed 4317 using the frozen preparation code and verify the canonical split hash:

`2ef72de30dff98ac4c67da1eb4f68ba0e40156953fa76ffaade7371c0eaa9484`

Expected counts are train 14,198, validation 3,011, and internal test 3,127, with no patient crossing partitions.

