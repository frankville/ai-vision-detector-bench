import hashlib
import os

import pytest

from visionbench.sources.identity import identify_source


def test_content_identity_is_independent_of_path(tmp_path):
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    data = b"clip bytes" * 200000
    first.write_bytes(data)
    second.write_bytes(data)
    identity = identify_source(str(first))
    assert identity.metadata() == identify_source(str(second)).metadata() == {
        "content_sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
    }
    assert str(tmp_path) not in str(identity.metadata())
    second.write_bytes(b"different clip")
    assert identify_source(str(second)).sha256 != identity.sha256


@pytest.mark.parametrize("url", ["rtsp://user:secret@host/live", "https://host/video.mp4", "file:///clip.mp4"])
def test_url_identity_stays_unknown_without_file_access(monkeypatch, url):
    monkeypatch.setattr("pathlib.Path.stat", lambda *a: pytest.fail("URL inspected as a file"))
    identity = identify_source(url)
    identity.verify()
    assert identity.metadata() == {"content_sha256": None, "size_bytes": None}


def test_nonregular_input_not_opened(tmp_path, monkeypatch):
    fifo = tmp_path / "input"
    os.mkfifo(fifo)
    monkeypatch.setattr(os, "open", lambda *a: pytest.fail("FIFO opened"))
    assert identify_source(str(fifo)).sha256 is None


def test_mutation_while_hashing_rejected(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"original")
    original = hashlib.sha256

    def mutate():
        clip.write_bytes(b"changed")
        return original()

    monkeypatch.setattr(hashlib, "sha256", mutate)
    with pytest.raises(ValueError, match="changed"):
        identify_source(str(clip))


def test_mutation_during_content_read_rejected(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"original")
    original = hashlib.sha256

    class MutatingHash:
        def __init__(self):
            self.digest = original()

        def update(self, data):
            clip.write_bytes(b"changed")
            self.digest.update(data)

    monkeypatch.setattr(hashlib, "sha256", MutatingHash)
    with pytest.raises(ValueError, match="changed while hashing"):
        identify_source(str(clip))


@pytest.mark.parametrize("operation", ["modify", "delete", "replace"])
def test_later_file_changes_rejected(tmp_path, operation):
    clip = tmp_path / "private-name.mp4"
    clip.write_bytes(b"original")
    identity = identify_source(str(clip))
    if operation == "modify":
        clip.write_bytes(b"changed")
    elif operation == "delete":
        clip.unlink()
    else:
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"original")
        replacement.replace(clip)
    with pytest.raises(ValueError, match="changed") as error:
        identity.verify()
    assert str(tmp_path) not in str(error.value)
    assert clip.name not in str(error.value)


def test_missing_file_error_has_no_path(tmp_path):
    with pytest.raises(ValueError, match="could not fingerprint") as error:
        identify_source(str(tmp_path / "private-name.mp4"))
    assert str(tmp_path) not in str(error.value)
