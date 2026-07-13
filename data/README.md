# Data

Nothing in this directory is committed (see `.gitignore`) — KAIST is ~35 GB.

Expected layout after `scripts/prepare_kaist.py` runs:

```
data/kaist/
  images/                 # setXX/VYYY/{visible,lwir}/IIIIII.jpg
  annotations/
    train/                # sanitized training annotations
    test/                 # improved test annotations
  splits/                 # the train/test image lists used by the literature
```

The annotation set is not a detail. The raw 2015 annotations contain known errors; the field
standardized on a sanitized training set and improved test annotations. Using anything else makes
your MR⁻² incomparable to every published number. See `docs/LITERATURE.md` for the provenance and
download sources.
