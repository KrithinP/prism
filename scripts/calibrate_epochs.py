"""Epoch calibration: how long should the from-scratch recipe train?

The LR sweep used 8 epochs to rank learning rates -- a sweep artifact, not a recipe.
8 is almost certainly undertrained (YOLOX does ~300 from scratch), and an undertrained
baseline would make fusion look good for the wrong reason (Ch.8 lesson). So before the
5-seed baselines, we find where MR^-2 stops improving.

Method: ONE thermal-only run at the locked lr=0.01, trained long, checkpointing every
few epochs, then MR^-2 evaluated at each checkpoint to trace MR vs epochs.

Honesty caveat baked into the reading: the intermediate checkpoints are saved
mid-cosine-schedule (LR still high), so an epoch-k snapshot of a long run is a LOWER
BOUND on a properly-annealed k-epoch run. The trend is trustworthy; the absolute
intermediate values are pessimistic. The fully-annealed FINAL checkpoint is the real
"train this many epochs" number, directly comparable to the sweep's annealed 8-epoch
point (36.56% All).

Run:
  PYTHONPATH= .venv/bin/python scripts/calibrate_epochs.py --epochs 40 --save-every 8
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prism.evaluate_model import evaluate_checkpoint

OUT = Path("outputs/epoch_cal")
FIGS = Path("docs/figures")
SWEEP_8EP_ALL = 36.56           # the annealed 8-epoch reference from the LR sweep


def train_subprocess(root, modality, lr, epochs, save_every, batch_size):
    """Train in a separate process so a GPU fault can't kill this driver, and so the
    intermediate checkpoints on disk survive even if something later crashes."""
    cmd = [sys.executable, "-m", "prism.train",
           "--root", root, "--modality", modality, "--lr", str(lr),
           "--epochs", str(epochs), "--save-every", str(save_every),
           "--batch-size", str(batch_size), "--seed", "0", "--out", str(OUT)]
    print("  training:", " ".join(cmd))
    proc = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=""))
    if proc.returncode != 0:
        raise RuntimeError(f"training exited with code {proc.returncode}")


def checkpoint_epoch(p: Path, total_epochs: int) -> int:
    """thermal_seed0_e16.pt -> 16 ; thermal_seed0.pt -> total_epochs (final, annealed)."""
    stem = p.stem
    return int(stem.split("_e")[-1]) if "_e" in stem else total_epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/kaist-cvpr15")
    ap.add_argument("--modality", default="thermal", choices=["rgb", "thermal"])
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--save-every", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse existing checkpoints in outputs/epoch_cal (eval only)")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    if not a.skip_train:
        train_subprocess(a.root, a.modality, a.lr, a.epochs, a.save_every, a.batch_size)

    ckpts = sorted(OUT.glob(f"{a.modality}_seed0*.pt"),
                   key=lambda p: checkpoint_epoch(p, a.epochs))
    if not ckpts:
        print("no checkpoints found in", OUT); return

    rows = []
    for p in ckpts:
        ep = checkpoint_epoch(p, a.epochs)
        final = "_e" not in p.stem
        # full All/Day/Night only for the final (annealed) checkpoint; All for the rest
        splits = ("test", "test-day", "test-night") if final else ("test",)
        _, res = evaluate_checkpoint(str(p), a.root, a.modality, splits=splits,
                                     batch_size=a.batch_size)
        row = {"epochs": ep, "annealed": final,
               "mr_all": res["test"]["mr"] * 100,
               "mr_day": res.get("test-day", {}).get("mr", float("nan")) * 100,
               "mr_night": res.get("test-night", {}).get("mr", float("nan")) * 100}
        rows.append(row)
        tag = "annealed" if final else "mid-schedule"
        print(f"  epochs={ep:>3} ({tag}): MR^-2 All={row['mr_all']:.2f}%")

    rows.sort(key=lambda r: r["epochs"])
    csv_path = OUT / f"{a.modality}_epoch_cal.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["epochs", "annealed", "mr_all", "mr_day", "mr_night"])
        w.writeheader(); w.writerows(rows)

    # --- figure ---
    mid = [(r["epochs"], r["mr_all"]) for r in rows if not r["annealed"]]
    ann = [(r["epochs"], r["mr_all"]) for r in rows if r["annealed"]]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    if mid:
        ax.plot(*zip(*mid), "o--", color="#999", label="mid-schedule (lower bound)")
    ax.scatter([8], [SWEEP_8EP_ALL], color="#1f77b4", zorder=5,
               label=f"annealed 8ep (sweep): {SWEEP_8EP_ALL:.1f}%")
    if ann:
        ax.scatter(*zip(*ann), color="#d62728", zorder=5, s=70,
                   label=f"annealed {a.epochs}ep: {ann[0][1]:.1f}%")
    ax.set_xlabel("training epochs"); ax.set_ylabel("MR$^{-2}$ All (%, lower better)")
    ax.set_title(f"Epoch calibration ({a.modality}-only, from scratch, lr={a.lr})")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig_path = FIGS / f"epoch_calibration_{a.modality}.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight"); plt.close(fig)

    print(f"\nsaved {csv_path}")
    print(f"saved {fig_path}")
    if ann:
        delta = SWEEP_8EP_ALL - ann[0][1]
        verdict = ("MORE epochs clearly help -- 8 was undertrained"
                   if delta > 2 else "diminishing returns past ~8 epochs")
        print(f"\nannealed 8ep {SWEEP_8EP_ALL:.1f}%  ->  annealed {a.epochs}ep "
              f"{ann[0][1]:.1f}%   (Δ {delta:+.1f})  =>  {verdict}")


if __name__ == "__main__":
    main()
