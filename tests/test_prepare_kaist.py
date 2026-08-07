"""Dataset preparation must not extract archive members outside its destination."""

import io
import tarfile

import pytest

from scripts.prepare_kaist import extract_tar


def test_extract_tar_accepts_regular_files(tmp_path):
    archive = tmp_path / "safe.tar"
    payload = b"KAIST"
    with tarfile.open(archive, "w") as handle:
        member = tarfile.TarInfo("nested/frame.txt")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))

    destination = tmp_path / "data"
    extract_tar(archive, destination)

    assert (destination / "nested/frame.txt").read_bytes() == payload


def test_extract_tar_rejects_parent_traversal(tmp_path):
    archive = tmp_path / "traversal.tar"
    with tarfile.open(archive, "w") as handle:
        member = tarfile.TarInfo("../escape.txt")
        member.size = 1
        handle.addfile(member, io.BytesIO(b"x"))

    with pytest.raises(SystemExit, match="unsafe path"):
        extract_tar(archive, tmp_path / "data")

    assert not (tmp_path / "escape.txt").exists()


def test_extract_tar_rejects_external_symlink(tmp_path):
    archive = tmp_path / "symlink.tar"
    with tarfile.open(archive, "w") as handle:
        member = tarfile.TarInfo("nested/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../escape.txt"
        handle.addfile(member)

    with pytest.raises(SystemExit, match="unsafe tar member"):
        extract_tar(archive, tmp_path / "data")

    assert not (tmp_path / "escape.txt").exists()
