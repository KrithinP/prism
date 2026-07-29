"""Phase 1 baselines: single-camera detectors, 5 seeds each, with the noise floor.

These are the scores fusion must beat. For each modality (thermal, rgb) we train the
locked recipe (lr=0.01, ~16 epochs from scratch) across 5 seeds, evaluate MR^-2
All/Day/Night on the improved-test set, and report mean +/- std. The std is THE noise
floor: any later fusion claim needs |delta| > 2*std to count as real.

Crash-proof + resumable (same pattern as the lr sweep): each run trains in its own
subprocess (a GPU fault costs one run), an existing checkpoint is not retrained, and
results flush to CSV after every run.

Run:
  PYTHONPATH= .venv/bin/python scripts/run_baselines.py --epochs 16 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from prism.evaluate_model import evaluate_checkpoint

OUT = Path("outputs/baselines")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/kaist-cvpr15")
    ap.add_argument("--modalities", nargs="+", default=["thermal", "rgb"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "baselines.csv"
    rows = []

    def flush():
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["modality", "seed", "mr_all", "mr_day", "mr_night"])
            w.writeheader(); w.writerows(rows)

    for modality in a.modalities:
        for seed in a.seeds:
            out = OUT / f"{modality}_seed{seed}"
            ckpt = out / f"{modality}_seed{seed}.pt"
            print(f"\n{'='*70}\n  BASELINE {modality} seed {seed}\n{'='*70}")
            try:
                if ckpt.exists():
                    print(f"  [skip training] {ckpt} exists")
                else:
                    cmd = [sys.executable, "-m", "prism.train",
                           "--root", a.root, "--modality", modality, "--lr", str(a.lr),
                           "--epochs", str(a.epochs), "--batch-size", str(a.batch_size),
                           "--seed", str(seed), "--out", str(out)]
                    if subprocess.run(cmd, env=dict(os.environ, PYTHONPATH="")).returncode != 0:
                        raise RuntimeError("training subprocess failed")

                _, res = evaluate_checkpoint(str(ckpt), a.root, modality,
                                             batch_size=a.batch_size)
                row = {"modality": modality, "seed": seed,
                       "mr_all": res["test"]["mr"] * 100,
                       "mr_day": res["test-day"]["mr"] * 100,
                       "mr_night": res["test-night"]["mr"] * 100}
                rows.append(row)
                print(f"  -> {modality} seed {seed}: All={row['mr_all']:.2f}%  "
                      f"Day={row['mr_day']:.2f}%  Night={row['mr_night']:.2f}%")
            except Exception as e:
                print(f"  !! {modality} seed {seed} FAILED: {type(e).__name__}: {e}")
            flush()

    # --- noise floor: mean +/- std per modality ---
    summary = {}
    for modality in a.modalities:
        vals = {k: [r[k] for r in rows if r["modality"] == modality]
                for k in ("mr_all", "mr_day", "mr_night")}
        if not vals["mr_all"]:
            continue
        summary[modality] = {
            k: {"mean": statistics.mean(v),
                "std": statistics.pstdev(v) if len(v) > 1 else 0.0,
                "n": len(v)}
            for k, v in vals.items()}

    (OUT / "baselines_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*70}\n  NOISE FLOOR (mean +/- std over seeds, MR^-2 %)\n{'='*70}")
    print(f"{'modality':<10}{'All':>16}{'Day':>16}{'Night':>16}")
    for modality, s in summary.items():
        def cell(k): return f"{s[k]['mean']:.2f}+/-{s[k]['std']:.2f}"
        print(f"{modality:<10}{cell('mr_all'):>16}{cell('mr_day'):>16}{cell('mr_night'):>16}")
    print(f"\nsaved {csv_path}")
    print("The Night std is the bar every later claim must clear (2-sigma).")


if __name__ == "__main__":
    main()
