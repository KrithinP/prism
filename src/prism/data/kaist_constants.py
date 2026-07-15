"""KAIST protocol constants — the parts of the benchmark that are convention, not code.

These are scaffolding: they encode decisions the *field* made (which class is positive,
what "reasonable" means), not algorithm. Getting any of these wrong silently makes your
MR incomparable to every published number, so they live in one named place, documented,
rather than as magic numbers scattered through the dataset code.

Sources: Hwang et al. CVPR 2015; the standard KAIST evaluation protocol as used by
MSDS-RCNN / MBNet / ProbEn. See docs/LITERATURE.md 1.3-1.4.
"""

# --- Classes -----------------------------------------------------------------
# KAIST annotations carry four names. The protocol treats ONLY "person" as a
# positive to be detected. Everything else is an *ignore* region: a detection that
# lands on it is neither a true positive nor a false positive — it is discarded
# before scoring. This is why you must keep ignore boxes around at eval time rather
# than deleting them; they suppress false positives you'd otherwise be charged for.
POSITIVE_CLASSES = ("person",)
IGNORE_CLASSES = ("people", "cyclist", "person?")   # present in labels, never scored as positives
ALL_CLASSES = POSITIVE_CLASSES + IGNORE_CLASSES

# --- "Reasonable" evaluation subset ------------------------------------------
# The headline MR^-2 is reported on pedestrians that are large enough and not
# heavily hidden. Definition used across the literature:
REASONABLE_MIN_HEIGHT = 55      # pixels; pedestrians shorter than this are ignored
REASONABLE_MAX_OCCLUSION = 1    # occlusion 0 (none) or 1 (partial) kept; 2 (heavy) ignored

# Occlusion tag meanings, as stored in the XML <occlusion> field:
OCCLUSION_NONE = 0
OCCLUSION_PARTIAL = 1
OCCLUSION_HEAVY = 2

# --- Image geometry ----------------------------------------------------------
# Every KAIST frame is 640x512. Many summaries say 640x480 — trust the data, not
# the summaries (the XML <size> and the decoded JPGs both say 512).
IMG_WIDTH = 640
IMG_HEIGHT = 512

# --- On-disk layout ----------------------------------------------------------
# images/setXX/VYYY/{visible,lwir}/IZZZZZ.jpg  and a matching annotation XML.
# A split-file line looks like "set06/V000/I00019" (no modality, no extension).
VISIBLE_DIRNAME = "visible"
LWIR_DIRNAME = "lwir"
