"""Run the validation-only architecture matrix, resumably and fail closed.

This program has no official-test option and no test-split code path. It trains
on ``train-cv``, evaluates ``val-cv``, persists raw logs/JSON, and appends one
ledger row per attempt. Block C requires the validation-selected Block B winner
written by ``scripts/select_winner.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from prism.protocol import ARCH_EPOCHS, PROTOCOL_VERSION, RECIPE, SEEDS
from prism.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    acquire_run_lock,
    append_jsonl,
    artifact_record,
    atomic_write_json,
    environment_snapshot,
    new_attempt_id,
    read_jsonl,
    release_run_lock,
    run_logged_subprocess,
    unfinished_attempts,
    utc_now,
)

LEDGER = Path("outputs/ledger.jsonl")
OUT = Path("outputs/experiments")

ARMS = {
    "A1": {"label": "RGB baseline", "train": ["--modality", "rgb"]},
    "A2": {"label": "Thermal baseline", "train": ["--modality", "thermal"]},
    "A3": {"label": "RGB wide control", "train": ["--modality", "rgb", "--base-ch", "42"]},
    "A4": {"label": "Thermal wide control", "train": ["--modality", "thermal", "--base-ch", "42"]},
    "A5": {
        "label": "RGB+RGB capacity",
        "train": [
            "--modality",
            "fusion",
            "--fusion-family",
            "multi-scale",
            "--fuse-subset",
            "345",
            "--fusion-inputs",
            "rgb",
        ],
    },
    "A6": {
        "label": "T+T capacity",
        "train": [
            "--modality",
            "fusion",
            "--fusion-family",
            "multi-scale",
            "--fuse-subset",
            "345",
            "--fusion-inputs",
            "thermal",
        ],
    },
    "A7": {
        "label": "paired nested residual, thermal anchor at initialization",
        "train": [
            "--modality",
            "fusion",
            "--fusion-family",
            "multi-scale",
            "--fuse-subset",
            "345",
            "--anchor",
            "thermal",
            "--fusion-operator",
            "nested-residual",
        ],
    },
    "A8": {
        "label": "paired nested residual, RGB anchor at initialization",
        "train": [
            "--modality",
            "fusion",
            "--fusion-family",
            "multi-scale",
            "--fuse-subset",
            "345",
            "--anchor",
            "rgb",
            "--fusion-operator",
            "nested-residual",
        ],
    },
    "B1": {
        "label": "Early input concat",
        "train": ["--modality", "fusion", "--fusion-family", "early"],
    },
    "B2": {
        "label": "One-shot meet d3",
        "train": [
            "--modality",
            "fusion",
            "--fusion-family",
            "one-shot",
            "--meet-level",
            "3",
            "--anchor",
            "thermal",
        ],
    },
    "B3": {
        "label": "One-shot meet d4",
        "train": [
            "--modality",
            "fusion",
            "--fusion-family",
            "one-shot",
            "--meet-level",
            "4",
            "--anchor",
            "thermal",
        ],
    },
    "B4": {
        "label": "One-shot meet d5",
        "train": [
            "--modality",
            "fusion",
            "--fusion-family",
            "one-shot",
            "--meet-level",
            "5",
            "--anchor",
            "thermal",
        ],
    },
    "B5": {
        "label": "MS fuse {3}",
        "train": ["--modality", "fusion", "--fusion-family", "multi-scale", "--fuse-subset", "3"],
    },
    "B6": {
        "label": "MS fuse {4}",
        "train": ["--modality", "fusion", "--fusion-family", "multi-scale", "--fuse-subset", "4"],
    },
    "B7": {
        "label": "MS fuse {5}",
        "train": ["--modality", "fusion", "--fusion-family", "multi-scale", "--fuse-subset", "5"],
    },
    "B8": {
        "label": "MS fuse {3,4}",
        "train": ["--modality", "fusion", "--fusion-family", "multi-scale", "--fuse-subset", "34"],
    },
    "B9": {
        "label": "MS fuse {4,5}",
        "train": ["--modality", "fusion", "--fusion-family", "multi-scale", "--fuse-subset", "45"],
    },
    "B10": {
        "label": "MS fuse {3,4,5}",
        "train": ["--modality", "fusion", "--fusion-family", "multi-scale", "--fuse-subset", "345"],
    },
    # B11/B12 are post-training controls in run_late_fusion_matrix.py.
    "B13": {"label": "Winner with RGB anchor", "train": []},
    "C1": {"label": "op average", "train": ["--fusion-operator", "average"]},
    "C2": {"label": "op concat", "train": ["--fusion-operator", "concat"]},
    "C3": {"label": "op scalar-mixture", "train": ["--fusion-operator", "scalar-mixture"]},
    "C4": {"label": "op channel-mixture", "train": ["--fusion-operator", "channel-mixture"]},
    "C5": {"label": "op gated-residual", "train": ["--fusion-operator", "gated-residual"]},
    "C6": {"label": "op GAFF-inspired", "train": ["--fusion-operator", "gaff"]},
    "C7": {"label": "op DMAF-inspired", "train": ["--fusion-operator", "dmaf"]},
    "C8": {"label": "op learned alignment", "train": ["--fusion-operator", "alignment"]},
    "C9": {
        "label": "op bounded cross-attention",
        "train": ["--fusion-operator", "cross-attention"],
    },
    "C10": {
        "label": "op sample-adaptive quality gate",
        "train": ["--fusion-operator", "quality-gated"],
    },
    "C11": {
        "label": "cross-attention parameter-matched concat control",
        "train": ["--fusion-operator", "concat-attention-params"],
    },
    "C12": {
        "label": "alignment parameter-matched spatial concat control",
        "train": ["--fusion-operator", "concat-alignment-params"],
    },
    "D1": {"label": "winner + modality dropout p=0.2", "train": ["--modality-dropout-p", "0.2"]},
    "D2": {
        "label": "winner + thermal translation augmentation 8px",
        "train": ["--thermal-shift-px", "8"],
    },
    "D3": {
        "label": "winner + dropout and translation augmentation",
        "train": ["--modality-dropout-p", "0.2", "--thermal-shift-px", "8"],
    },
}
BLOCKS = {
    "A": [f"A{i}" for i in range(1, 9)],
    "B": [f"B{i}" for i in range(1, 11)],
    "C": [f"C{i}" for i in range(1, 13)],
    "D": [f"D{i}" for i in range(1, 4)],
}


def log_row(row: dict) -> None:
    append_jsonl(LEDGER, row)


def load_winning_args(path: str | None) -> list[str]:
    if not path:
        raise SystemExit("this arm requires --winning-graph-manifest from scripts/select_winner.py")
    manifest = json.loads(Path(path).read_text())
    if manifest.get("status") != "selected" or not isinstance(manifest.get("args"), list):
        raise SystemExit(f"invalid winning-graph manifest: {path}")
    args = list(manifest["args"])
    if "--modality" not in args or args[args.index("--modality") + 1] != "fusion":
        raise SystemExit("winning graph is not a fusion model")
    return args


def without_option(args: list[str], option: str) -> list[str]:
    result = []
    index = 0
    while index < len(args):
        if args[index] == option:
            index += 2
        else:
            result.append(args[index])
            index += 1
    return result


def set_option(args: list[str], option: str, value: str) -> list[str]:
    result = without_option(args, option)
    return [*result, option, value]


def effective_args(arm_id: str, winner_path: str | None) -> list[str]:
    if arm_id.startswith("C"):
        graph = without_option(load_winning_args(winner_path), "--fusion-operator")
        return graph + ARMS[arm_id]["train"]
    if arm_id == "B13":
        graph = load_winning_args(winner_path)
        if "--fusion-family" in graph and graph[graph.index("--fusion-family") + 1] == "early":
            raise SystemExit("B13 anchor sensitivity is undefined when early fusion wins")
        return set_option(graph, "--anchor", "rgb")
    if arm_id.startswith("D"):
        return load_winning_args(winner_path) + ARMS[arm_id]["train"]
    return list(ARMS[arm_id]["train"])


def option_value(args: list[str], option: str, default: str) -> str:
    return args[args.index(option) + 1] if option in args else default


def config_hash(args: list[str], source_hash: str) -> str:
    payload = {
        "args": args,
        "split": "train-cv",
        "recipe": RECIPE,
        "protocol_version": PROTOCOL_VERSION,
        "source_fingerprint": source_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_fingerprint() -> str:
    """Hash executable study source so stale checkpoints cannot be reused."""
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "src" / "prism").rglob("*.py"))
    files += [root / "configs" / "architecture.yaml", Path(__file__).resolve()]
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def checkpoint_matches(
    path: Path, expected_hash: str, expected_source: str, seed: int, expected_epoch: int
) -> bool:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    # A corrupt or incompatible checkpoint is not resumable; retrain that run.
    except Exception:
        return False
    return (
        state.get("run_args_hash") == expected_hash
        and state.get("source_fingerprint") == expected_source
        and state.get("protocol_version") == PROTOCOL_VERSION
        and state.get("seed") == seed
        and state.get("epoch") == expected_epoch
    )


def find_resume_checkpoint(
    directory: Path,
    modality: str,
    seed: int,
    expected_hash: str,
    expected_source: str,
    final_epoch: int,
) -> Path | None:
    candidates = []
    for path in directory.glob(f"{modality}_seed{seed}_e*.pt"):
        suffix = path.stem.rsplit("_e", 1)[-1]
        if not suffix.isdigit():
            continue
        epoch = int(suffix)
        if epoch >= final_epoch:
            raise RuntimeError(f"invalid periodic checkpoint epoch {epoch}: {path}")
        if not checkpoint_matches(path, expected_hash, expected_source, seed, epoch):
            raise RuntimeError(f"periodic checkpoint metadata does not match this run: {path}")
        candidates.append((epoch, path))
    return max(candidates)[1] if candidates else None


def provenance_inputs(root: str) -> list[Path]:
    candidates = [
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("configs/architecture.yaml"),
        Path("splits/train.txt"),
        Path("splits/val.txt"),
        Path("splits/split_manifest.json"),
        Path("external/evaluator_manifest.json"),
        Path(root) / "prism_data_manifest.json",
    ]
    return [path for path in candidates if path.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=list(ARMS))
    parser.add_argument("--block", choices=list(BLOCKS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--root", default="data/raw/kaist-cvpr15")
    parser.add_argument("--lr", type=float, default=RECIPE["lr"])
    parser.add_argument("--epochs", type=int, default=ARCH_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=RECIPE["batch_size"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--winning-graph-manifest", default=None)
    args = parser.parse_args()

    if bool(args.arm) == bool(args.block):
        raise SystemExit("pass exactly one of --arm or --block")
    if len(args.seeds) != len(set(args.seeds)):
        raise SystemExit("seed list contains duplicates")
    if args.limit and len(args.seeds) != 1:
        raise SystemExit("smoke mode (--limit) takes exactly one seed")
    if not args.limit:
        expected = (RECIPE["lr"], ARCH_EPOCHS, RECIPE["batch_size"])
        actual = (args.lr, args.epochs, args.batch_size)
        if actual != expected:
            raise SystemExit(
                f"non-smoke architecture runs are frozen at lr/epochs/batch={expected}, "
                f"got {actual}"
            )
        if tuple(args.seeds) != SEEDS and args.block in ("A", "B", "C", "D"):
            raise SystemExit(f"core blocks require all predeclared seeds {SEEDS}")

    arm_ids = [args.arm] if args.arm else BLOCKS[args.block]
    base_env = dict(os.environ, PYTHONPATH="", PYTHONUNBUFFERED="1")
    source_hash = source_fingerprint()
    failures = 0
    for arm_id in arm_ids:
        arm = ARMS[arm_id]
        train_args = effective_args(arm_id, args.winning_graph_manifest)
        modality = option_value(train_args, "--modality", "fusion")
        input_policy = option_value(train_args, "--fusion-inputs", "paired")
        run_hash = config_hash(train_args, source_hash)
        for seed in args.seeds:
            env = dict(base_env, PYTHONHASHSEED=str(seed))
            out = OUT / f"{arm_id}_s{seed}"
            out.mkdir(parents=True, exist_ok=True)
            checkpoint = out / f"{modality}_seed{seed}.pt"
            attempt_id = new_attempt_id(f"{arm_id}_s{seed}")
            run_lock = acquire_run_lock(OUT / ".locks" / f"{arm_id}_s{seed}.lock")
            for stale_id, stale in unfinished_attempts(
                read_jsonl(LEDGER), arm=arm_id, seed=seed
            ).items():
                log_row(
                    {
                        **stale,
                        "event": "attempt_abandoned",
                        "timestamp": utc_now(),
                        "status": "abandoned",
                        "error": "prior process ended without a terminal event",
                        "superseded_by": attempt_id,
                        "abandoned_attempt_id": stale_id,
                    }
                )
            attempt_dir = out / "attempts" / attempt_id
            attempt_dir.mkdir(parents=True, exist_ok=False)
            attempt_started = time.monotonic()
            val_json = attempt_dir / "val_metrics.json"
            history_path = attempt_dir / "training_history.jsonl"
            train_stdout = attempt_dir / "train.stdout.log"
            train_stderr = attempt_dir / "train.stderr.log"
            val_stdout = attempt_dir / "val.stdout.log"
            val_stderr = attempt_dir / "val.stderr.log"
            eval_command = [
                sys.executable,
                "-m",
                "prism.evaluate_model",
                "--ckpt",
                str(checkpoint),
                "--split",
                "val-cv",
                "--root",
                args.root,
                "--input-policy",
                input_policy,
                "--json-out",
                str(val_json),
                "--raw-output-dir",
                str(attempt_dir / "raw_frames"),
                "--parent-manifest",
                str(attempt_dir / "attempt_manifest.json"),
            ]
            train_command = [
                sys.executable,
                "-m",
                "prism.train",
                "--root",
                args.root,
                "--lr",
                str(args.lr),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--split",
                "train-cv",
                "--seed",
                str(seed),
                "--out",
                str(out),
                "--history-path",
                str(history_path),
                "--run-id",
                f"{arm_id}_s{seed}",
                "--attempt-id",
                attempt_id,
                "--run-args-hash",
                run_hash,
                "--source-fingerprint",
                source_hash,
                "--save-every",
                str(RECIPE["checkpoint_every_epochs"]),
                "--val-every",
                str(RECIPE["validation_every_epochs"]),
                *train_args,
            ]
            if args.limit:
                train_command += ["--limit", str(args.limit), "--positives-only"]

            started_at = utc_now()
            start_manifest = {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "kind": "experiment_attempt",
                "attempt_id": attempt_id,
                "started_at_utc": started_at,
                "arm": arm_id,
                "label": arm["label"],
                "seed": seed,
                "protocol_version": PROTOCOL_VERSION,
                "recipe": RECIPE,
                "train_args": train_args,
                "args_hash": run_hash,
                "source_fingerprint": source_hash,
                "runner_argv": sys.argv,
                "subprocess_environment_overrides": {
                    "PYTHONPATH": "",
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONHASHSEED": str(seed),
                },
                "planned_commands": {
                    "train": train_command,
                    "validation": eval_command,
                },
                "paths": {
                    "output_dir": str(out.resolve()),
                    "attempt_dir": str(attempt_dir.resolve()),
                    "checkpoint": str(checkpoint.resolve()),
                    "validation_json": str(val_json.resolve()),
                    "training_history": str(history_path.resolve()),
                },
                "environment": environment_snapshot(
                    Path(__file__).resolve().parents[1],
                    input_artifacts=provenance_inputs(args.root),
                ),
            }
            start_manifest_path = attempt_dir / "attempt_manifest.json"
            atomic_write_json(start_manifest_path, start_manifest)
            start_manifest_artifact = artifact_record(start_manifest_path)
            base_row = {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "attempt_id": attempt_id,
                "arm": arm_id,
                "label": arm["label"],
                "seed": seed,
                "args": train_args,
                "args_hash": run_hash,
                "source_fingerprint": source_hash,
                "protocol_version": PROTOCOL_VERSION,
                "checkpoint": str(checkpoint),
                "input_policy": input_policy,
                "split": "val-cv",
            }
            log_row(
                {
                    **base_row,
                    "event": "attempt_started",
                    "timestamp": started_at,
                    "status": "started",
                    "attempt_manifest": start_manifest_artifact,
                }
            )

            terminal = {
                **base_row,
                "event": "attempt_finished",
                "timestamp": None,
                "val": None,
                "status": "failed",
                "training": None,
                "phases": {},
                "attempt_manifest": start_manifest_artifact,
            }
            try:
                if checkpoint.exists():
                    if not checkpoint_matches(checkpoint, run_hash, source_hash, seed, args.epochs):
                        raise RuntimeError(
                            f"existing checkpoint metadata does not match this run: {checkpoint}"
                        )
                    terminal["training"] = "reused"
                    terminal["phases"]["training"] = {
                        "status": "reused",
                        "checkpoint": artifact_record(checkpoint),
                    }
                else:
                    resume = find_resume_checkpoint(
                        out, modality, seed, run_hash, source_hash, args.epochs
                    )
                    if resume is not None:
                        train_command += ["--resume", str(resume)]
                        terminal["resumed_from"] = artifact_record(resume)
                    train_result = run_logged_subprocess(
                        train_command,
                        train_stdout,
                        train_stderr,
                        env=env,
                    )
                    terminal["phases"]["training"] = train_result
                    if train_result["returncode"] != 0:
                        raise RuntimeError(
                            "training subprocess failed with exit code "
                            f"{train_result['returncode']}"
                        )
                    if not checkpoint_matches(checkpoint, run_hash, source_hash, seed, args.epochs):
                        raise RuntimeError(
                            f"training did not produce a valid final checkpoint: {checkpoint}"
                        )
                    terminal["training"] = "completed"
                    terminal["phases"]["training"]["checkpoint"] = artifact_record(checkpoint)

                val_result = run_logged_subprocess(
                    eval_command,
                    val_stdout,
                    val_stderr,
                    env=env,
                )
                terminal["phases"]["validation"] = val_result
                if val_result["returncode"] != 0:
                    raise RuntimeError(
                        f"validation subprocess failed with exit code {val_result['returncode']}"
                    )
                metrics = json.loads(val_json.read_text())["splits"]["val-cv"]
                if metrics["mr_all"] is None:
                    raise RuntimeError("validation produced no primary MR")
                terminal["val"] = metrics
                terminal["validation_artifact"] = artifact_record(val_json)
                terminal["checkpoint_artifact"] = artifact_record(checkpoint)
                terminal["status"] = "completed"
            # One failed arm must be recorded without losing the remaining matrix.
            except Exception as error:
                failures += 1
                terminal["error"] = f"{type(error).__name__}: {error}"
                terminal["traceback"] = traceback.format_exc()
                print(f"{arm_id} seed {seed}: {terminal['error']}", file=sys.stderr)
            terminal["timestamp"] = utc_now()
            terminal["duration_seconds"] = time.monotonic() - attempt_started
            result_path = attempt_dir / "attempt_result.json"
            atomic_write_json(result_path, terminal)
            terminal["result_manifest"] = artifact_record(result_path)
            log_row(terminal)
            release_run_lock(run_lock)
            print(f"{arm_id} seed {seed}: {terminal['status']} ({attempt_id})")

    print(f"rows appended to {LEDGER}")
    if failures:
        raise SystemExit(f"{failures} matrix run(s) failed")


if __name__ == "__main__":
    main()
