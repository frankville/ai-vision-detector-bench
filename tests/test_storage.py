import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from visionbench import storage


def test_creates_nested_directory_and_preserves_collisions(tmp_path):
    directory = tmp_path / "nested" / "results"
    first = storage.write_json(str(directory), "run", {"value": 1})
    original = first.read_bytes()
    second = storage.write_json(directory, "run", {"value": 2})
    assert first != second
    assert first.read_bytes() == original
    assert json.loads(second.read_text()) == {"value": 2}
    assert original.endswith(b"\n")
    assert sorted(path.suffix for path in directory.iterdir()) == [".json", ".json"]


def test_concurrent_writers_publish_complete_distinct_results(tmp_path):
    def write(index):
        return storage.write_json(tmp_path, "run", {"index": index, "payload": "x" * 10000})

    with ThreadPoolExecutor(max_workers=8) as executor:
        paths = list(executor.map(write, range(24)))
    assert len(set(paths)) == 24
    assert {json.loads(path.read_text())["index"] for path in paths} == set(range(24))
    assert all(json.loads(path.read_text())["payload"] == "x" * 10000 for path in paths)
    assert set(tmp_path.iterdir()) == set(paths)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), object()])
def test_serialization_failure_leaves_existing_result_untouched(tmp_path, value):
    existing = storage.write_json(tmp_path, "run", {"value": 1})
    original = existing.read_bytes()
    with pytest.raises((ValueError, TypeError)):
        storage.write_json(tmp_path, "run", {"value": value})
    assert existing.read_bytes() == original
    assert list(tmp_path.iterdir()) == [existing]


@pytest.mark.parametrize("operation", ["link", "fsync"])
def test_publication_or_flush_failure_cleans_temporary_file(tmp_path, monkeypatch, operation):
    existing = storage.write_json(tmp_path, "run", {"value": 1})
    original = existing.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    monkeypatch.setattr(storage.os, operation, fail)
    with pytest.raises(OSError, match="injected failure"):
        storage.write_json(tmp_path, "run", {"value": 2})
    assert existing.read_bytes() == original
    assert list(tmp_path.iterdir()) == [existing]


@pytest.mark.parametrize("stem", ["", ".", "..", "../run", "nested/run", "nested\\run"])
def test_rejects_path_stems(tmp_path, stem):
    with pytest.raises(ValueError, match="stem"):
        storage.write_json(tmp_path, stem, {})
    assert not list(tmp_path.iterdir())
