"""Figures for the LR-sweep chapter (textbook Ch.8).

  fig 1  lr_sweep_mr.png   -- MR^-2 (All/Day/Night) vs learning rate, with the
                             divergence cliff (0.02) and the unstable point (0.04) marked.
  fig 2  lr_divergence.png -- the objectness loss of the diverged lr=0.02 run, parsed
                             from the real training log, so "divergence" is a picture.

Run: PYTHONPATH= .venv/bin/python scripts/make_lr_figures.py
"""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("docs/figures"); OUT.mkdir(parents=True, exist_ok=True)

# --- sweep results (from outputs/sweeps/thermal_lrsweep.csv) ---
LR = [0.00125, 0.0025, 0.005, 0.01]
ALL = [43.21, 38.67, 38.98, 36.56]
DAY = [50.94, 44.71, 44.81, 43.47]
NIGHT = [27.35, 26.11, 26.89, 22.61]


def fig_sweep():
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(LR, ALL, "-o", label="All", color="#1f77b4", lw=2, ms=7)
    ax.plot(LR, DAY, "-s", label="Day", color="#ff7f0e", lw=2, ms=6)
    ax.plot(LR, NIGHT, "-^", label="Night", color="#2ca02c", lw=2, ms=6)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (log scale)")
    ax.set_ylabel("MR$^{-2}$  (%, lower is better)")
    ax.set_title("Thermal-only from-scratch: MR$^{-2}$ vs learning rate")

    # annotate the winner
    ax.annotate("winner: 0.01\n(36.56% All)", xy=(0.01, 36.56), xytext=(0.0045, 33.5),
                arrowprops=dict(arrowstyle="->"), fontsize=9)
    # the two points we could not plot, because they don't produce a usable model
    ax.axvspan(0.014, 0.05, color="red", alpha=0.07)
    ax.text(0.019, 48, "0.02 diverges\n0.04 unstable\n(no usable MR)",
            color="#a00", fontsize=8.5, ha="center")
    ax.set_xlim(0.001, 0.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="test subset")
    fig.tight_layout()
    p = OUT / "lr_sweep_mr.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    return p


def fig_divergence(log="outputs/sweep_thermal.log"):
    steps, obj = [], []
    if Path(log).exists():
        for line in Path(log).read_text().splitlines():
            m = re.search(r"s(\d+)/25080 lr0\.020.*obj([0-9.]+)", line)
            if m:
                steps.append(int(m.group(1))); obj.append(float(m.group(2)))
    if not steps:                       # fallback to the recorded trace
        steps = [800, 1000, 1100, 1200, 1260, 1280]
        obj = [110, 12.4, 256.7, 402.1, 663.7, 988.8]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(steps, obj, "-o", color="#d62728", ms=4)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("objectness loss (log scale)")
    ax.set_title("What divergence looks like: lr = 0.02, objectness runs away")
    ax.axhline(300, color="gray", ls="--", lw=1)
    ax.text(steps[0], 340, "our divergence guard (EMA > 300)", color="gray", fontsize=8)
    ax.annotate("device-side assert\nkilled the process here",
                xy=(steps[-1], obj[-1]), xytext=(steps[-1] - 300, obj[-1] * 0.3),
                arrowprops=dict(arrowstyle="->"), fontsize=9, color="#a00")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = OUT / "lr_divergence.png"
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    return p


if __name__ == "__main__":
    print("saved", fig_sweep())
    print("saved", fig_divergence())
