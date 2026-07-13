# Data

Nothing in this directory is committed (see `.gitignore`) — KAIST is ~36 GB.

## Status (2026-07-13)

**Preview set: DOWNLOADED and md5-verified.** `data/raw/kaist-cvpr15-preview/`
- 3,200 aligned pairs — `set04/V001` (a **night** sequence), `images/set04/V001/{visible,lwir}/*.jpg`
- 3,200 **sanitized** annotations — `annotations-xml-new-sanitized/set04/V001/*.xml`
- **All the official split files** — `imageSets/` ← the most valuable part; this is the protocol
- macOS `._*` AppleDouble junk was present in the tarball and has been deleted. Re-deleting after
  any fresh extract is required, or the dataloader will try to read them as images.

**Full set (36.3 GB): NOT downloaded.** Google Drive returns its "too many accesses" quota block
for the large file (`gdown 1sBcAmFqNJmNMBZdMtKmO2X4BRjKPyKMc`). Two routes:
1. **Official direct mirror (recommended):** http://ircv-nas.hanyang.ac.kr:5000/sharing/WfRD94z0t
   — a Synology share; it needs a browser session, so `curl`/`wget` get a 400. Open it in a
   browser, download `kaist-cvpr15.tar` into `data/raw/`, then verify:
   `md5sum -c kaist-cvpr15.tar.md5` (expected `fcfc7cfcc1237ad414af64b972f8dc97`, already saved here).
2. Retry `gdown` after the Drive quota resets (typically ~24 h).

**Do NOT use third-party re-uploads** (e.g. the Hugging Face `richidubey/KAIST-*` mirrors). They
are partial, train-only, and unclear on annotations. The annotation set is the one thing that must
be canonical — see `docs/LITERATURE.md` §1.3.

## Protocol verification already done

The official split files were checked against the published protocol and they match:

| Split file | Lines | Meaning |
|---|---:|---|
| `test-all-20.txt` | **2,252** | the standard test set — matches every published paper |
| `test-day-20.txt` | 1,455 | day subset |
| `test-night-20.txt` | 797 | night subset (1,455 + 797 = 2,252 ✓) |
| `train-all-02.txt` | 25,086 | every 2nd frame of the training sets |
| `train-all-20.txt` | 2,500 | every 20th frame |

**Open question for Phase 0:** the literature quotes a **7,601**-image sanitized training set, but
`train-all-02.txt` has 25,086 entries. The likely explanation is that the sanitized set drops
frames containing no pedestrians (25,086 → 7,601). **Verify this before locking the protocol** —
if the train-set size is wrong, every number in the study is off, and it will not be obvious.

Note the annotation XML reports image size **640×512**, not the 640×480 that many papers and
summaries state. Trust the data.

## Expected layout after `scripts/prepare_kaist.py`

```
data/kaist/
  images/                 # setXX/VYYY/{visible,lwir}/IIIIII.jpg
  annotations/            # sanitized train, improved test
  splits/                 # the imageSets files above
```

The annotation set is not a detail. The raw 2015 annotations contain known errors; the field
standardized on sanitized training annotations (Li et al./MSDS-RCNN) and improved test annotations
(Liu et al. 2018). Using anything else makes MR⁻² incomparable to every published number — and the
cleanup was worth ~4 MR points, which is more than any fusion operator in the literature.
