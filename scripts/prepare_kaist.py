"""Prepare the exact KAIST protocol used by PRISM, then fail-closed audit it.

Sources:
  * official full image archive: KAIST benchmark Google Drive ID;
  * sanitized train labels: MSDS-RCNN release Google Drive ID;
  * improved test labels: pinned MLPD ``KAIST_annotation.json``.

No test images or labels are used for training/splitting. The improved test JSON
is installed only so a later frozen release can be evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from prism.provenance import atomic_write_json

KAIST_ARCHIVE_ID = "1sBcAmFqNJmNMBZdMtKmO2X4BRjKPyKMc"
KAIST_MD5_ID = "1KQ9lZXX9nfZuhbqBUEdlU2Yjx3nzeg5c"
SANITIZED_TRAIN_ID = "1mKIzND2046aXHknzr28LJHvFH_6Ssa_q"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(file_id: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "gdown", file_id, "-O", str(output)], check=True)


def verify_md5(archive: Path, checksum_file: Path) -> str:
    expected = checksum_file.read_text().split()[0].lower()
    digest = hashlib.md5()  # nosec B324 - integrity check mandated by dataset publisher
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise SystemExit(f"KAIST archive MD5 mismatch: expected {expected}, got {actual}")
    return actual


def extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        # Refuse path traversal instead of trusting an archive-wide extract.
        root = destination.resolve()
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise SystemExit(f"unsafe path in archive: {member.name}")
        try:
            handle.extractall(destination, filter="data")
        except tarfile.FilterError as error:
            raise SystemExit(f"unsafe tar member: {error}") from error


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/kaist-cvpr15")
    parser.add_argument("--downloads", default="data/downloads")
    parser.add_argument(
        "--download",
        action="store_true",
        help="download missing archives (otherwise existing files are required)",
    )
    parser.add_argument(
        "--skip-audits",
        action="store_true",
        help="conversion-only debug option; not valid before training",
    )
    args = parser.parse_args()

    root = Path(args.root)
    downloads = Path(args.downloads)
    archive = downloads / "kaist-cvpr15.tar"
    md5_file = downloads / "kaist-cvpr15.tar.md5"
    sanitized = downloads / "sanitized_annotations.tar"
    for file_id, path in (
        (KAIST_ARCHIVE_ID, archive),
        (KAIST_MD5_ID, md5_file),
        (SANITIZED_TRAIN_ID, sanitized),
    ):
        if not path.exists():
            if not args.download:
                raise SystemExit(f"missing {path}; rerun with --download")
            download(file_id, path)

    archive_md5 = verify_md5(archive, md5_file)
    if not (root / "images").is_dir():
        extract_tar(archive, root.parent)
    if not (root / "images").is_dir() or not (root / "imageSets").is_dir():
        raise SystemExit(f"archive did not produce expected {root}/{{images,imageSets}} layout")

    sanitized_dir = downloads / "sanitized_annotations"
    if not sanitized_dir.exists():
        extract_tar(sanitized, sanitized_dir)

    evaluator_json = Path("external/MLPD-eval/evaluation_script/KAIST_annotation.json")
    if not evaluator_json.exists():
        run([sys.executable, "scripts/pin_mlpd_eval.py", "--clone"])

    run(
        [
            sys.executable,
            "scripts/convert_kaist_bbgt.py",
            "--root",
            str(root),
            "--train-src",
            str(sanitized_dir),
            "--test-json",
            str(evaluator_json),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/make_splits.py",
            "--root",
            str(root),
            "--out-dir",
            "splits",
        ]
    )
    if not args.skip_audits:
        run(
            [
                sys.executable,
                "scripts/audit_dataset.py",
                "--root",
                str(root),
                "--split",
                "train",
                "--out",
                "outputs/audit/train.json",
            ]
        )
        run(
            [
                sys.executable,
                "scripts/audit_dataset.py",
                "--root",
                str(root),
                "--split",
                "test",
                "--out",
                "outputs/audit/test.json",
            ]
        )

    audit_reports = {}
    if not args.skip_audits:
        for split in ("train", "test"):
            report = Path("outputs/audit") / f"{split}.json"
            audit_reports[split] = {
                "path": str(report.resolve()),
                "sha256": sha256(report),
            }

    manifest = {
        "schema_version": 1,
        "prepared_at": datetime.now(UTC).isoformat(),
        "root": str(root.resolve()),
        "sources": {
            "kaist_archive": {
                "google_drive_id": KAIST_ARCHIVE_ID,
                "md5": archive_md5,
                "sha256": sha256(archive),
            },
            "sanitized_train": {
                "google_drive_id": SANITIZED_TRAIN_ID,
                "sha256": sha256(sanitized),
            },
            "improved_test_json": {
                "path": str(evaluator_json.resolve()),
                "sha256": sha256(evaluator_json),
            },
        },
        "split_manifest": {
            "path": str(Path("splits/split_manifest.json").resolve()),
            "sha256": sha256(Path("splits/split_manifest.json")),
        },
        "protocol_counts": {
            "train": sum(
                1
                for line in (root / "imageSets/train-sanitized.txt").read_text().splitlines()
                if line.strip()
            ),
            "test": sum(
                1
                for line in (root / "imageSets/test-all-20.txt").read_text().splitlines()
                if line.strip()
            ),
        },
        "audit_reports": audit_reports,
        "audits_completed": not args.skip_audits,
    }
    output = root / "prism_data_manifest.json"
    atomic_write_json(output, manifest)
    print(f"prepared KAIST protocol: {output}")


if __name__ == "__main__":
    main()
