"""Generate the explanatory figures for the evaluator (textbook Ch.3 / docs).

    python scripts/make_eval_figures.py

Writes to outputs/figures/:
  fig1_mr_fppi_curve.png   -- the worked example: threshold sweep -> curve -> 9 points
  fig2_reasonable_filter.png -- real KAIST heights, and what the 55px cutoff ignores
  fig3_ignore_breakdown.png  -- why each ground-truth box is scored or ignored
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prism.data.kaist import PairedKAIST
from prism.data.kaist_constants import REASONABLE_MAX_OCCLUSION, REASONABLE_MIN_HEIGHT
from prism.eval.mr import REF_FPPI, log_average_miss_rate

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def fig1_curve():
    """The worked example from the explanation, drawn."""
    # 5 images, 5 pedestrians; detector finds 4, raises 2 false alarms.
    labels = np.array([1, 0, 1, 1, 0, 1])  # TP FP TP TP FP TP
    scores = np.array([0.95, 0.85, 0.80, 0.60, 0.55, 0.40])
    n_pos, n_images = 5, 5

    order = np.argsort(-scores)
    lab = labels[order]
    tp, fp = np.cumsum(lab == 1), np.cumsum(lab == 0)
    miss, fppi = 1.0 - tp / n_pos, fp / n_images

    mrs = []
    for r in REF_FPPI:
        ok = np.where(fppi <= r)[0]
        mrs.append(miss[ok[-1]] if len(ok) else 1.0)
    mr2 = log_average_miss_rate(labels, scores, n_pos, n_images)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # left: the sweep as a staircase
    ax1.step(fppi, miss, where="post", lw=2, color="#1f77b4")
    ax1.scatter(fppi, miss, zorder=3, color="#1f77b4")
    for i, (f, m, s) in enumerate(zip(fppi, miss, scores[order], strict=True)):
        ax1.annotate(
            f"keep top {i + 1}\n(score>={s:.2f})",
            (f, m),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=6.5,
        )
    ax1.set_xlabel("FPPI  (false positives per image)")
    ax1.set_ylabel("miss rate")
    ax1.set_title("Walking down the sorted detections\n= turning the confidence dial loose")
    ax1.grid(alpha=0.3)
    ax1.set_ylim(-0.05, 1.0)

    # right: log-x with the nine reference budgets
    ax2.step(
        np.clip(fppi, 1e-3, None), miss, where="post", lw=2, color="#1f77b4", label="detector curve"
    )
    for r, m in zip(REF_FPPI, mrs, strict=True):
        ax2.axvline(r, color="grey", lw=0.6, ls=":")
        ax2.scatter([r], [m], color="#d62728", zorder=4, s=28)
    ax2.scatter([], [], color="#d62728", label="miss rate at the 9 FPPI budgets")
    ax2.set_xscale("log")
    ax2.set_xlim(8e-3, 1.3)
    ax2.set_ylim(-0.05, 1.0)
    ax2.set_xlabel("FPPI (log scale)")
    ax2.set_ylabel("miss rate")
    ax2.set_title(f"MR$^{{-2}}$ = geometric mean of the 9 red dots = {mr2:.3f}\n(lower is better)")
    ax2.legend(fontsize=7, loc="lower left")
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(OUT / "fig1_mr_fppi_curve.png")
    plt.close(fig)
    return mr2


def _collect_gt(max_frames=3200):
    ds = PairedKAIST("data/raw/kaist-cvpr15-preview")
    heights, classes, occl = [], [], []
    for i in range(min(len(ds), max_frames)):
        s = ds[i]
        b = s["boxes"].numpy()
        if len(b) == 0:
            continue
        heights.append(b[:, 3] - b[:, 1])
        classes.append(s["classes"].numpy())
        occl.append(s["occlusion"].numpy())
    return (np.concatenate(heights), np.concatenate(classes), np.concatenate(occl))


def fig2_and_3():
    h, c, o = _collect_gt()

    scored = (c == 1) & (h >= REASONABLE_MIN_HEIGHT) & (o <= REASONABLE_MAX_OCCLUSION)

    # --- fig2: height distribution vs the 55px cutoff ---
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, max(200, h.max()), 60)
    ax.hist(h[c == 1], bins=bins, color="#9ecae1", label="'person' boxes")
    ax.axvline(
        REASONABLE_MIN_HEIGHT,
        color="#d62728",
        lw=2,
        label=f"Reasonable cutoff = {REASONABLE_MIN_HEIGHT}px",
    )
    frac = (h[c == 1] < REASONABLE_MIN_HEIGHT).mean()
    ax.set_title(
        f"KAIST pedestrian heights (preview set)\n"
        f"{frac * 100:.0f}% of 'person' boxes are BELOW the cutoff -> ignored, not scored"
    )
    ax.set_xlabel("bounding-box height (pixels)")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_reasonable_filter.png")
    plt.close(fig)

    # --- fig3: why each box is scored or ignored ---
    reasons = {
        "SCORED\n(person, >=55px,\nocc<=1)": int(scored.sum()),
        "ignored:\nnot 'person'": int((c == 0).sum()),
        "ignored:\ntoo small": int(((c == 1) & (h < REASONABLE_MIN_HEIGHT)).sum()),
        "ignored:\nheavily occluded": int(
            ((c == 1) & (h >= REASONABLE_MIN_HEIGHT) & (o > REASONABLE_MAX_OCCLUSION)).sum()
        ),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2ca02c", "#bbbbbb", "#bbbbbb", "#bbbbbb"]
    ax.bar(list(reasons), list(reasons.values()), color=colors)
    for i, v in enumerate(reasons.values()):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    total = sum(reasons.values())
    scored = next(iter(reasons.values()))
    ax.set_title(
        f"Of {total} ground-truth boxes, only {scored} "
        f"({scored / total * 100:.0f}%) are actually scored\n"
        f"the rest are IGNORE regions: they absorb detections, penalty-free"
    )
    ax.set_ylabel("ground-truth boxes")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_ignore_breakdown.png")
    plt.close(fig)
    return reasons, total


if __name__ == "__main__":
    mr2 = fig1_curve()
    reasons, total = fig2_and_3()
    print(f"fig1: worked-example MR^-2 = {mr2:.3f}")
    print(f"fig2/3: {total} GT boxes -> {next(iter(reasons.values()))} scored")
    for k, v in reasons.items():
        print(f"   {k.replace(chr(10), ' ')}: {v}")
    print(f"figures in {OUT}/")
