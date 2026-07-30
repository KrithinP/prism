"""Phase 2 — the fusion-depth sweep: where should the two cameras meet?

Trains concat fusion at each backbone depth alone (dark3 / dark4 / dark5) and compares
against the F-cat-345 anchor (all depths, already trained in Phase 1). Operator fixed to
concat, recipe fixed (lr=0.01, 16 epochs) -- only the fuse depth varies, so any
difference is attributable to depth.

Expected (from the literature): on CLEAN/aligned KAIST the table is fairly FLAT -- depth
barely matters when the cameras line up. That flatness is the point: it is the aligned
baseline against which Phase 4 breaks alignment, where shallow (dark3) should collapse and
deep (dark5) should survive.

Crash-proof + resumable (per-config subprocess, skip-if-checkpoint-exists, flush per run).

Run: PYTHONPATH= .venv/bin/python scripts/sweep_fusion_stage.py --seeds 0 1 2
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

OUT = Path("outputs/fusion_stage")
# name -> fuse-stages string. 345 (all) is the Phase-1 anchor, re-used from outputs/fusion.
CONFIGS = [("dark3", "3"), ("dark4", "4"), ("dark5", "5")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/kaist-cvpr15")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "fusion_stage.csv"
    rows = []

    def flush():
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["config", "seed", "mr_all", "mr_day", "mr_night"])
            w.writeheader(); w.writerows(rows)

    for name, stages in CONFIGS:
        for seed in a.seeds:
            out = OUT / f"{name}_seed{seed}"
            ckpt = out / f"fusion_seed{seed}.pt"
            print(f"\n{'='*70}\n  FUSION STAGE {name} (fuse@{stages}) seed {seed}\n{'='*70}")
            try:
                if ckpt.exists():
                    print(f"  [skip training] {ckpt} exists")
                else:
                    cmd = [sys.executable, "-m", "prism.train",
                           "--root", a.root, "--modality", "fusion",
                           "--fuse-stages", stages, "--lr", str(a.lr),
                           "--epochs", str(a.epochs), "--batch-size", str(a.batch_size),
                           "--seed", str(seed), "--out", str(out)]
                    if subprocess.run(cmd, env=dict(os.environ, PYTHONPATH="")).returncode != 0:
                        raise RuntimeError("training subprocess failed")
                _, res = evaluate_checkpoint(str(ckpt), a.root, "fusion", batch_size=a.batch_size)
                row = {"config": name, "seed": seed,
                       "mr_all": res["test"]["mr"] * 100,
                       "mr_day": res["test-day"]["mr"] * 100,
                       "mr_night": res["test-night"]["mr"] * 100}
                rows.append(row)
                print(f"  -> {name} seed {seed}: All={row['mr_all']:.2f}%  "
                      f"Day={row['mr_day']:.2f}%  Night={row['mr_night']:.2f}%")
            except Exception as e:
                print(f"  !! {name} seed {seed} FAILED: {type(e).__name__}: {e}")
            flush()

    # summary per config
    summary = {}
    for name, _ in CONFIGS:
        vals = {k: [r[k] for r in rows if r["config"] == name]
                for k in ("mr_all", "mr_day", "mr_night")}
        if vals["mr_all"]:
            summary[name] = {k: {"mean": statistics.mean(v),
                                 "std": statistics.pstdev(v) if len(v) > 1 else 0.0}
                             for k, v in vals.items()}
    (OUT / "fusion_stage_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*70}\n  FUSION-DEPTH SWEEP (mean+/-std, MR^-2 %)\n{'='*70}")
    print(f"{'config':<10}{'All':>16}{'Day':>16}{'Night':>16}")
    print(f"{'345 (anchor)':<10}{'27.7+/-1.9':>16}{'32.7+/-2.3':>16}{'17.5+/-1.4':>16}")
    for name, s in summary.items():
        c = lambda k: f"{s[k]['mean']:.1f}+/-{s[k]['std']:.1f}"
        print(f"{name:<10}{c('mr_all'):>16}{c('mr_day'):>16}{c('mr_night'):>16}")
    print(f"\nsaved {csv_path}")


if __name__ == "__main__":
    main()
