"""The protocol splits must load the right counts and keep train/test disjoint by set."""
import pytest

from prism.data.kaist import kaist_split

ROOT = "data/raw/kaist-cvpr15"


@pytest.mark.parametrize("split,expected", [
    ("train", 25086), ("test", 2252), ("test-day", 1455), ("test-night", 797)])
def test_split_counts_match_protocol(split, expected):
    ds = kaist_split(ROOT, split)
    assert len(ds) == expected, f"{split}: got {len(ds)}, expected {expected}"


def test_train_and_test_use_disjoint_sets():
    train = kaist_split(ROOT, "train")
    test = kaist_split(ROOT, "test")
    train_sets = {fid.split('/')[0] for fid in train.ids}
    test_sets = {fid.split('/')[0] for fid in test.ids}
    assert train_sets == {"set00", "set01", "set02", "set03", "set04", "set05"}
    assert test_sets <= {"set06", "set07", "set08", "set09", "set10", "set11"}
    assert train_sets.isdisjoint(test_sets)      # the whole point: no scene leakage
