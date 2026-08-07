"""The training-ready layer must batch images and hand the loss a usable target list."""

import torch

from prism.data.detection import detection_collate


class _FakeBase:
    """Two frames: one with a positive + an ignore box, one empty."""

    def __len__(self):
        return 2

    def __getitem__(self, i):
        if i == 0:
            return {
                "rgb": torch.zeros(3, 512, 640),
                "thermal": torch.zeros(3, 512, 640),
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 60.0], [0.0, 0.0, 5.0, 5.0]]),
                "classes": torch.tensor([1, 0]),  # one person, one ignore
                "occlusion": torch.tensor([0, 0]),
            }
        return {
            "rgb": torch.zeros(3, 512, 640),
            "thermal": torch.zeros(3, 512, 640),
            "boxes": torch.zeros(0, 4),
            "classes": torch.zeros(0, dtype=torch.long),
            "occlusion": torch.zeros(0, dtype=torch.long),
        }


def test_dataset_and_collate_shapes():
    from prism.data.detection import DetectionDataset

    ds = DetectionDataset(_FakeBase(), modality="rgb")

    img0, t0 = ds[0]
    assert img0.shape == (3, 512, 640)
    assert img0.max() <= 1.0  # normalized to [0,1]
    assert t0.shape == (2, 5)  # positive and ignore both preserved
    assert t0[0, 0] == 0.0  # class index remapped to 0
    assert t0[1, 0] == -1.0  # evaluator ignore region

    _, t1 = ds[1]
    assert t1.shape == (0, 5)  # empty frame -> empty target

    imgs, targets = detection_collate([ds[0], ds[1]])
    assert imgs.shape == (2, 3, 512, 640)  # images stacked
    assert isinstance(targets, list) and len(targets) == 2  # targets stay a list
