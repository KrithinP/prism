"""Stages 2-3 of the evaluator: the miss-rate/FPPI curve, and the log-average MR^-2.

Feed it the matched labels/scores from every frame (concatenated), how many real
pedestrians there were in total, and how many images. Get back one number.
See textbook Ch.3, Stages 2-3.

The mental model (worked example in the chat transcript / Ch.3):
  * Sorting detections high-score-first and walking down the list IS the
    confidence-threshold sweep -- each step "keeps" one more detection.
  * At each step: miss_rate = 1 - TP/n_pos, and fppi = FP/n_images.
  * Read the miss rate at nine fixed false-alarm budgets, then geometric-mean them.
"""
from __future__ import annotations

import numpy as np

# The nine FPPI reference points: log-spaced from 1e-2 to 1e0.
REF_FPPI = np.logspace(-2.0, 0.0, num=9)


def log_average_miss_rate(labels, scores, n_pos, n_images):
    """labels: (K,) 1=TP / 0=FP across ALL frames. scores: (K,) their confidences.
    n_pos: total scored pedestrians. n_images: number of test images."""
    labels = np.asarray(labels, np.int64).reshape(-1)
    scores = np.asarray(scores, np.float32).reshape(-1)

    if n_pos == 0:
        return float("nan")               # nothing to detect -> undefined
    if len(labels) == 0:
        return 1.0                        # no detections at all -> miss everyone

    # Lower the threshold = walk detections high-score to low-score, accumulating.
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels == 1)           # true positives so far
    fp = np.cumsum(labels == 0)           # false positives so far

    miss_rate = 1.0 - tp / n_pos          # fraction of pedestrians still missed
    fppi = fp / n_images                  # false positives per image

    # At each reference FPPI, read off the miss rate at that false-alarm budget.
    mrs = []
    for r in REF_FPPI:
        ok = np.where(fppi <= r)[0]       # operating points within this budget
        mrs.append(miss_rate[ok[-1]] if len(ok) else 1.0)
    mrs = np.clip(np.asarray(mrs), 1e-10, 1.0)   # clip so log() is safe

    # geometric mean = exp(mean(log(.)))
    return float(np.exp(np.mean(np.log(mrs))))
