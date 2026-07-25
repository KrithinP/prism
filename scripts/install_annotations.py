"""Install and VERIFY the real KAIST annotation sets.

Background (decisions log 2026-07-24): the two annotation dirs on disk,
`annotations-xml-new` and `annotations-xml-new-sanitized`, are byte-identical -- we
hold one annotation set under two names, so the train=sanitized / test=improved mapping
in kaist.py is a no-op. Our numbers are internally valid but not comparable to the
published leaderboard until the real sets are installed:

  * sanitized TRAIN annotations (Li et al. / MSDS-RCNN)
        https://li-chengyang.github.io/home/MSDS-RCNN/  (Google Drive link on the page)
  * improved TEST annotations (Liu et al. 2018)
        request form linked from the same page (gated -- may take days)

This script does not download (the sources are a Drive link + a Google Form). It takes
a directory you have already downloaded and unpacked, checks it is really different from
what we have, sanity-checks structure and counts against the official split lists, and
only then installs it -- so a wrong or truncated download fails loudly instead of
silently poisoning every result.

Usage:
  # dry-run verification (default): does NOT modify anything
  PYTHONPATH= .venv/bin/python scripts/install_annotations.py \
      --which train --src ~/Downloads/sanitized_annotations

  # actually install (backs up the current dir first)
  PYTHONPATH= .venv/bin/python scripts/install_annotations.py \
      --which train --src ~/Downloads/sanitized_annotations --install
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path("data/raw/kaist-cvpr15")

# which annotation dir each logical set feeds, and which split list bounds it
TARGETS = {
    "train": {"dir": "annotations-xml-new-sanitized", "list": "train-all-02.txt",
              "sets": {f"set{n:02d}" for n in range(0, 6)}},
    "test":  {"dir": "annotations-xml-new",           "list": "test-all-20.txt",
              "sets": {f"set{n:02d}" for n in range(6, 12)}},
}


def find_ann_root(src: Path) -> Path:
    """Accept either the annotation root itself or a parent that contains it.
    We locate it by finding the shallowest dir that holds setXX/ subdirs of XMLs."""
    if any((src / f"set{n:02d}").is_dir() for n in range(12)):
        return src
    for p in sorted(src.rglob("set00")):
        if p.is_dir() and any(p.rglob("*.xml")):
            return p.parent
    raise SystemExit(f"could not find a setXX/ annotation tree under {src}")


def read_ids(list_file: Path) -> list[str]:
    return [ln.strip() for ln in list_file.read_text().splitlines() if ln.strip()]


def verify(which: str, src: Path) -> dict:
    """Return a report dict; raise SystemExit on any disqualifying problem."""
    spec = TARGETS[which]
    current = ROOT / spec["dir"]
    new_root = find_ann_root(src)
    ids = read_ids(ROOT / "imageSets" / spec["list"])

    report = {"which": which, "new_root": str(new_root), "current": str(current),
              "n_expected": len(ids)}

    # 1) every frame the split list names must have an XML in the new tree
    missing = [fid for fid in ids
               if not (new_root / f"{fid.rsplit('/', 1)[0]}" / f"{fid.rsplit('/', 1)[1]}.xml").exists()]
    report["missing"] = len(missing)
    report["missing_examples"] = missing[:5]

    # 2) it must be DIFFERENT from what we already have (else it's the same duplicate)
    same = 0
    checked = 0
    for fid in ids[:400]:
        rel = f"{fid.rsplit('/', 1)[0]}/{fid.rsplit('/', 1)[1]}.xml"
        a, b = new_root / rel, current / rel
        if a.exists() and b.exists():
            checked += 1
            if filecmp.cmp(a, b, shallow=False):
                same += 1
    report["checked"] = checked
    report["identical_to_current"] = same
    report["fraction_identical"] = (same / checked) if checked else 1.0

    # 3) XMLs must parse and contain plausible pedestrian labels
    names = set()
    parse_fail = 0
    for fid in ids[:200]:
        rel = f"{fid.rsplit('/', 1)[0]}/{fid.rsplit('/', 1)[1]}.xml"
        p = new_root / rel
        if not p.exists():
            continue
        try:
            for o in ET.parse(p).getroot().findall("object"):
                names.add(o.findtext("name"))
        except ET.ParseError:
            parse_fail += 1
    report["parse_failures"] = parse_fail
    report["label_names"] = sorted(n for n in names if n)

    # --- verdicts ---
    problems = []
    if missing:
        problems.append(f"{len(missing)} of {len(ids)} listed frames have no XML "
                        f"(e.g. {missing[:3]})")
    if parse_fail:
        problems.append(f"{parse_fail} XMLs failed to parse")
    if "person" not in names:
        problems.append(f"no 'person' label found in sample (names: {sorted(names)})")
    if report["fraction_identical"] > 0.98:
        problems.append("new annotations are ~identical to the current ones -- this is "
                        "the same set again, not the real download")
    report["problems"] = problems
    return report


def print_report(r: dict):
    print(f"\n=== verify {r['which']} annotations ===")
    print(f"  source tree : {r['new_root']}")
    print(f"  target dir  : {r['current']}")
    print(f"  frames in split list : {r['n_expected']}")
    print(f"  missing XMLs         : {r['missing']}  {r['missing_examples'] or ''}")
    print(f"  sampled {r['checked']} vs current: {r['identical_to_current']} identical "
          f"({r['fraction_identical']*100:.1f}%)")
    print(f"  parse failures       : {r['parse_failures']}")
    print(f"  label names in sample: {r['label_names']}")
    if r["problems"]:
        print("  VERDICT: NOT OK")
        for p in r["problems"]:
            print(f"    - {p}")
    else:
        print("  VERDICT: OK -- looks like a genuine, different, valid annotation set")


def install(which: str, src: Path):
    spec = TARGETS[which]
    current = ROOT / spec["dir"]
    new_root = find_ann_root(src)
    backup = current.with_name(current.name + ".dup-backup")
    if current.exists() and not backup.exists():
        print(f"  backing up current {current.name} -> {backup.name}")
        current.rename(backup)
    print(f"  installing {new_root} -> {current}")
    shutil.copytree(new_root, current, dirs_exist_ok=True)
    print("  done. Re-run verify (or `make test`) to confirm split counts still hold.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["train", "test"])
    ap.add_argument("--src", required=True, help="downloaded+unpacked annotation dir")
    ap.add_argument("--install", action="store_true",
                    help="actually install (default is verify-only dry run)")
    a = ap.parse_args()

    r = verify(a.which, Path(a.src).expanduser())
    print_report(r)
    if r["problems"]:
        print("\nRefusing to install: fix the download and re-run verification.")
        sys.exit(1)
    if a.install:
        install(a.which, Path(a.src).expanduser())
    else:
        print("\nDry run only. Re-run with --install to apply.")


if __name__ == "__main__":
    main()
