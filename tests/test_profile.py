"""Dynamic MAC accounting uses explicit, documented conventions."""

import torch
from torch import nn

from scripts.profile_checkpoint import count_dynamic_macs


def test_conv_mac_count():
    model = nn.Conv2d(3, 4, 3, padding=1, bias=False)
    inputs = (torch.zeros(1, 3, 8, 8),)
    assert count_dynamic_macs(model, inputs) == 1 * 4 * 8 * 8 * 3 * 3 * 3


def test_linear_mac_count():
    model = nn.Linear(5, 7, bias=False)
    inputs = (torch.zeros(3, 5),)
    assert count_dynamic_macs(model, inputs) == 3 * 7 * 5
