"""Soham's LR sweep: find the from-scratch learning rate before locking the recipe.

From scratch there is no pretrained model whose known-good LR we can borrow, so the
rate is an empirical unknown and must be measured. This trains one short run per
candidate LR (single seed -- the 5-seed protocol applies to the FINAL experiments,
not to recipe selection) and reports MR^-2 on the test split for each.

The winner becomes the fixed LR for every experiment in the study. Selection uses a
SINGLE seed on purpose: we are choosing a hyperparameter, not claiming a result.

Run:
  PYTHONPATH= .venv/bin/python scripts/sweep_lr.py --modality thermal --epochs 8
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from prism.train import train
from prism.evaluate_model import evaluate_checkpoint

DEFAULT_LRS = [0.04, 0.02, 0.01, 0.005, 0.0025]
OUT = Path("outputs/sweeps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/kaist-cvpr15")
    ap.add_argument("--modality", default="thermal", choices=["rgb", "thermal"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="optional: train on first N frames (faster, rougher sweep)")
    ap.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUT / f"{a.modality}_lrsweep"
    csv_path = OUT / f"{a.modality}_lrsweep.csv"
    rows = []

    def flush():
        """Write results after EVERY lr, so a later crash cannot lose earlier runs."""
        ordered = sorted(rows, key=lambda r: r["mr_all"])   # lower MR^-2 is better
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["lr", "mr_all", "mr_day", "mr_night"])
            w.writeheader(); w.writerows(ordered)
        (OUT / f"{a.modality}_lrsweep.json").write_text(json.dumps(ordered, indent=2))
        return ordered

    for lr in a.lrs:
        tag = f"lr{lr:g}"
        print(f"\n{'='*70}\n  LR SWEEP: {a.modality}  lr={lr:g}  epochs={a.epochs}\n{'='*70}")
        ckpt = ckpt_dir / tag / f"{a.modality}_seed{a.seed}.pt"
        try:
            if ckpt.exists():
                # Resume-friendly: an already-trained lr is not retrained (40 min each).
                print(f"  [skip training] checkpoint exists: {ckpt}")
            else:
                # Train in a SEPARATE PROCESS. A CUDA device-side assert corrupts the
                # context for the whole process, so an in-process crash at one lr kills
                # every later lr too (it did, 2026-07-24). Isolation makes each lr
                # independent: a diverged run costs that run only.
                cmd = [sys.executable, "-m", "prism.train",
                       "--root", a.root, "--modality", a.modality,
                       "--lr", str(lr), "--epochs", str(a.epochs),
                       "--batch-size", str(a.batch_size), "--seed", str(a.seed),
                       "--out", str(ckpt_dir / tag)]
                if a.limit:
                    cmd += ["--limit", str(a.limit)]
                env = dict(os.environ, PYTHONPATH="")
                proc = subprocess.run(cmd, env=env)
                if proc.returncode != 0:
                    raise RuntimeError(f"training exited with code {proc.returncode} "
                                       f"(diverged or crashed -- see output above)")

            _, res = evaluate_checkpoint(str(ckpt), a.root, a.modality,
                                         batch_size=a.batch_size)
            row = {"lr": lr,
                   "mr_all": res["test"]["mr"],
                   "mr_day": res["test-day"]["mr"],
                   "mr_night": res["test-night"]["mr"]}
            rows.append(row)
            print(f"  -> lr={lr:g}  MR^-2 All={row['mr_all']*100:.2f}%  "
                  f"Day={row['mr_day']*100:.2f}%  Night={row['mr_night']*100:.2f}%")
        except Exception as e:
            # One bad lr must not throw away the hours already spent on the others.
            print(f"  !! lr={lr:g} FAILED: {type(e).__name__}: {e}")
        flush()

    rows = flush()
    if not rows:
        print("\nNo lr completed successfully.")
        return

    print(f"\n{'='*70}\n  RESULT ({a.modality}, seed {a.seed}, {a.epochs} epochs)\n{'='*70}")
    print(f"{'lr':>10}{'MR^-2 All':>12}{'Day':>10}{'Night':>10}")
    for r in rows:
        print(f"{r['lr']:>10g}{r['mr_all']*100:>11.2f}%{r['mr_day']*100:>9.2f}%"
              f"{r['mr_night']*100:>9.2f}%")
    best = rows[0]
    print(f"\nBEST lr = {best['lr']:g}  (MR^-2 All {best['mr_all']*100:.2f}%)")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
