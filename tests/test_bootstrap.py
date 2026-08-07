"""Sequence grouping keeps temporally correlated frames together."""

from scripts.bootstrap_comparison import group_frames


def test_group_frames_uses_set_and_video():
    frames = [
        {"id": "set00/V000/I00000"},
        {"id": "set00/V001/I00000"},
        {"id": "set00/V000/I00001"},
    ]
    grouped = group_frames(frames)
    assert [row["id"] for row in grouped["set00/V000"]] == [
        "set00/V000/I00000",
        "set00/V000/I00001",
    ]
    assert len(grouped["set00/V001"]) == 1
