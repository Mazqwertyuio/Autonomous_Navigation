import hashlib
import io

import pytest

import download_dronet_v3 as setup


def test_download_is_atomic_and_checks_hash(tmp_path, monkeypatch):
    content = b"verified test checkpoint"
    monkeypatch.setitem(setup.CHECKPOINTS, "test", {
        "filename": "test.pth", "sha256": hashlib.sha256(content).hexdigest(),
    })
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(content))
    target = setup.download_checkpoint("test", tmp_path)
    assert target.read_bytes() == content
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("Redownloaded"))
    assert setup.download_checkpoint("test", tmp_path) == target


def test_download_rejects_corruption_without_leaving_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setitem(setup.CHECKPOINTS, "test", {"filename": "test.pth", "sha256": "0" * 64})
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="SHA-256"):
        setup.download_checkpoint("test", tmp_path)
    assert list(tmp_path.iterdir()) == []
