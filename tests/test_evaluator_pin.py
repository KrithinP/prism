"""The evaluator pin must be deterministic and exclude generated cache files."""

from scripts.pin_mlpd_eval import EXPECTED_COMMIT, build_manifest, sha256_file


def test_evaluator_manifest_excludes_python_cache(tmp_path) -> None:
    evaluation = tmp_path / "evaluation_script"
    cache = evaluation / "__pycache__"
    cache.mkdir(parents=True)
    source = evaluation / "evaluation_script.py"
    source.write_text("print('reference')\n")
    (cache / "evaluation_script.cpython-311.pyc").write_bytes(b"generated")

    manifest = build_manifest(tmp_path, EXPECTED_COMMIT)

    assert manifest["schema_version"] == 1
    assert manifest["commit_sha"] == EXPECTED_COMMIT
    assert manifest["files"] == {"evaluation_script/evaluation_script.py": sha256_file(source)}
