"""Pin the MLPD KAIST evaluation bundle (TODO.md 3.1) and record its SHA.

Clones the released evaluation implementation into external/MLPD-eval (the
untouched reference copy), records the commit SHA + file hashes in
external/evaluator_manifest.json, and prints the bundle's bundled-result
cross-check table. The local evaluator is then certified against this pinned
copy (scripts/parity_check.py), never against a moving remote branch.

Run:  PYTHONPATH= .venv/bin/python scripts/pin_mlpd_eval.py
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from prism.provenance import atomic_write_json

URL = "https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection.git"
EXPECTED_COMMIT = "8e583929a97b9f959634efa69174435ad0252a57"
DEST = Path("external/MLPD-eval")
MANIFEST = Path("external/evaluator_manifest.json")

# The repository's bundled published results, used as cross-check targets (TODO 3.3).
# MLPD_result.txt's exact identity is verified by its output, not assumed.
BUNDLED = {
    "CIAN": (14.12, 14.77, 11.13),
    "MSDS-RCNN": (11.34, 10.53, 12.94),
    "AR-CNN": (9.34, 9.94, 8.38),
    "MBNet": (8.13, 8.28, 7.86),
    "MLPD-VGG16": (7.58, 7.95, 6.95),
    "MLPD-ResNet50": (7.61, 8.36, 6.35),
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def clone_pinned_reference() -> None:
    """Fetch exactly the audited commit without depending on the remote HEAD."""
    if DEST.exists():
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {URL} at {EXPECTED_COMMIT}")
    subprocess.run(["git", "init", "--quiet", str(DEST)], check=True)
    subprocess.run(["git", "-C", str(DEST), "remote", "add", "origin", URL], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(DEST),
            "fetch",
            "--quiet",
            "--depth",
            "1",
            "origin",
            EXPECTED_COMMIT,
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(DEST), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
        check=True,
    )


def current_head() -> str:
    return subprocess.run(
        ["git", "-C", str(DEST), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_manifest(root: Path, commit: str) -> dict:
    eval_dir = root / "evaluation_script"
    files = sorted(
        path
        for path in eval_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    return {
        "schema_version": 1,
        "generator": "scripts/pin_mlpd_eval.py",
        "repository": URL,
        "commit_sha": commit,
        "files": {str(path.relative_to(root)): sha256_file(path) for path in files},
        "bundled_reference_results": BUNDLED,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--clone",
        action="store_true",
        help="actually clone; without it only the manifest is (re)built",
    )
    a = ap.parse_args()

    if not DEST.exists() and not a.clone:
        raise SystemExit(f"missing {DEST}; rerun with --clone")
    if a.clone:
        clone_pinned_reference()

    head = current_head()
    if head != EXPECTED_COMMIT:
        raise SystemExit(
            f"{DEST} is at {head}, expected audited commit {EXPECTED_COMMIT}; "
            "remove the fetched directory and rerun with --clone"
        )
    print(f"pinned at {head}")

    manifest = build_manifest(DEST, head)
    atomic_write_json(MANIFEST, manifest)
    print(f"manifest: {MANIFEST} ({len(manifest['files'])} files hashed)")
    print("\nbundled reference MR^-2 (All/Day/Night) to reproduce:")
    for name, (a_, d, n) in BUNDLED.items():
        print(f"  {name:<14} {a_:<8} {d:<8} {n}")


if __name__ == "__main__":
    main()
