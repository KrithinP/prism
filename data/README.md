# KAIST Data for PRISM

Data is not committed. A full preparation uses about 36 GB plus extracted files.

## Exact sources

Official KAIST image archive:

```text
page:            https://soonminhwang.github.io/rgbt-ped-detection/
Google Drive ID: 1sBcAmFqNJmNMBZdMtKmO2X4BRjKPyKMc
MD5 file ID:     1KQ9lZXX9nfZuhbqBUEdlU2Yjx3nzeg5c
```

MSDS-RCNN sanitized training annotations:

```text
project:         https://li-chengyang.github.io/home/MSDS-RCNN/
Google Drive ID: 1mKIzND2046aXHknzr28LJHvFH_6Ssa_q
expected frames: 7,601
```

Improved test annotations come from the pinned MLPD evaluator:

```text
external/MLPD-eval/evaluation_script/KAIST_annotation.json
expected frames: 2,252
```

Do not substitute original labels, KAIST-Paired labels, `train-all-02`, or a
third-party image subset while retaining the same experiment name. They define
different protocols.

## One command

From repository root:

```sh
make prepare-kaist
```

This executes `scripts/prepare_kaist.py --download`, which:

1. Downloads missing archives.
2. Verifies the official image archive MD5.
3. Safely extracts archives.
4. Fetches the MLPD evaluator at its exact audited commit if absent.
5. Converts sanitized train and improved test labels to the loader XML schema.
6. Requires exactly 7,601 train and 2,252 test IDs.
7. Freezes a sequence-disjoint train/validation split.
8. Audits every requested frame, pair, decode, dimension, XML, and box.
9. Writes `prism_data_manifest.json` with source, split, and audit hashes.

Then run:

```sh
make certify-evaluator
make preflight
```

## Expected layout

```text
data/raw/kaist-cvpr15/
  images/
    setXX/VYYY/visible/IZZZZZ.jpg
    setXX/VYYY/lwir/IZZZZZ.jpg
  imageSets/
    train-sanitized.txt
    test-all-20.txt
    test-day-20.txt
    test-night-20.txt
  annotations-sanitized-train/
    setXX/VYYY/IZZZZZ.xml
  annotations-improved-test/
    setXX/VYYY/IZZZZZ.xml
  prism_data_manifest.json

splits/
  train.txt
  val.txt
  split_manifest.json

outputs/audit/
  train.json
  test.json
```

## Geometry

The official page summarizes the dataset as 640x480. The released JPEGs and
improved annotation canvas consumed here are 640x512. The audit verifies decoded
shape `(512,640)` for every frame and fails on disagreement.

## Classes and ignore regions

Only exact `person` with no explicit ignore flag is positive. Group, cyclist,
uncertain, or explicit-ignore regions remain in the target as ignore boxes.
They suppress negative objectness anchors during training and absorb detections
under evaluator rules.

## Manual download fallback

If Google Drive is quota-limited, the official project currently lists a browser
NAS mirror:

<http://ircv-nas.hanyang.ac.kr:5000/sharing/WfRD94z0t>

Place the image archive, MD5 file, and sanitized annotation archive under
`data/downloads/`, then run:

```sh
PYTHONPATH= .venv/bin/python scripts/prepare_kaist.py
```

The script still verifies the image archive and all protocol counts.
