"""bbGt conversion must honor both class labels and explicit ignore flags."""

from pathlib import Path

from scripts.convert_kaist_bbgt import parse_bbgt


def test_bbgt_explicit_ignore_field_is_parsed(tmp_path: Path):
    annotation = tmp_path / "set00_V000_I00000.txt"
    annotation.write_text(
        "% bbGt version=3\nperson 1 2 30 60 0 1 2 30 60 1 0\nperson 5 6 30 60 0 5 6 30 60 0 0\n"
    )
    objects = parse_bbgt(str(annotation))
    assert objects[0]["ignore"] is True
    assert objects[1]["ignore"] is False
