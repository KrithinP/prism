"""Controlled input interventions must be deterministic and geometrically exact."""

import numpy as np
import torch

from scripts.evaluate_interventions import IntervenedDataset, write_raw_frames


class _Frame:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "id": "set00/V000/I00000",
            "rgb": torch.full((3, 8, 10), 100, dtype=torch.uint8),
            "thermal": torch.full((3, 8, 10), 200, dtype=torch.uint8),
            "boxes": torch.zeros(0, 4),
            "classes": torch.zeros(0, dtype=torch.long),
            "occlusion": torch.zeros(0, dtype=torch.long),
        }


_STATS = {
    "rgb": {"mean": [0.1, 0.2, 0.3]},
    "thermal": {"mean": [0.4, 0.5, 0.6]},
}


def test_mean_intervention_uses_train_statistics():
    sample = IntervenedDataset(_Frame(), "thermal_mean", _STATS)[0]
    assert sample["rgb"].unique().tolist() == [100]
    assert sample["thermal"][:, 0, 0].tolist() == [102, 127, 153]


def test_noise_intervention_is_deterministic_per_frame():
    dataset = IntervenedDataset(_Frame(), "thermal_noise_0.10", _STATS)
    assert torch.equal(dataset[0]["thermal"], dataset[0]["thermal"])


def test_severity_conditions_and_joint_null_are_exact():
    brightness = IntervenedDataset(_Frame(), "rgb_brightness_0.50", _STATS)[0]
    assert brightness["rgb"].unique().tolist() == [50]
    joint = IntervenedDataset(_Frame(), "both_zero", _STATS)[0]
    assert joint["rgb"].sum() == 0 and joint["thermal"].sum() == 0


def test_shift_intervention_has_no_wraparound():
    base = _Frame()
    frame = base[0]
    frame["thermal"].zero_()
    frame["thermal"][:, 3, 0] = 255

    class Changed(_Frame):
        def __getitem__(self, index):
            return frame

    shifted = IntervenedDataset(Changed(), "thermal_shift_x_m8", _STATS)[0]
    assert shifted["thermal"].sum() == 0


def test_raw_frame_artifact_round_trips_variable_length_rows(tmp_path):
    frames = [
        {
            "id": "a",
            "det_boxes": np.asarray([[1, 2, 3, 4]], np.float32),
            "det_scores": np.asarray([0.8], np.float32),
            "gt_boxes": np.asarray([[5, 6, 7, 8]], np.float32),
            "gt_classes": np.asarray([1], np.int64),
            "gt_occlusion": np.asarray([0], np.int64),
        },
        {
            "id": "b",
            "det_boxes": np.zeros((0, 4), np.float32),
            "det_scores": np.zeros(0, np.float32),
            "gt_boxes": np.zeros((0, 4), np.float32),
            "gt_classes": np.zeros(0, np.int64),
            "gt_occlusion": np.zeros(0, np.int64),
        },
    ]
    path = tmp_path / "raw.npz"
    report = write_raw_frames(frames, path)
    record = np.load(path)
    assert record["frame_ids"].tolist() == ["a", "b"]
    assert record["detection_offsets"].tolist() == [0, 1, 1]
    assert record["gt_offsets"].tolist() == [0, 1, 1]
    assert report["detections"] == 1 and report["ground_truth"] == 1
