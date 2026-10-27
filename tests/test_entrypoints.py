from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from visionbench import cli
from visionbench.web import app as web

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("command", ["run", "sweep"])
@pytest.mark.parametrize("flags", [
    ["--precision", "int8"], ["--backend", "onnxruntime"],
    ["--device", "cpu", "--precision", "fp16"],
    ["--device", "mps", "--precision", "amp-fp16"],
])
def test_cli_rejects_unsupported_requests(monkeypatch, command, flags):
    monkeypatch.setattr(cli, "run", Mock(side_effect=AssertionError("started")))
    monkeypatch.setattr(cli, "sweep", Mock(side_effect=AssertionError("started")))
    response = CliRunner().invoke(cli.app, [command, "-s", "missing.mp4", *flags])
    assert response.exit_code == 2
    assert "Invalid value" in response.output
    cli.run.assert_not_called()
    cli.sweep.assert_not_called()


@pytest.mark.parametrize("command", ["run", "sweep"])
def test_cli_forwards_default_execution_settings(monkeypatch, tmp_path, result, command):
    captured = []

    def fake_run(config):
        captured.append(config)
        return result

    def fake_sweep(config, keys, on_result, on_error, repeat=1, on_attempt=None):
        captured.append(config)
        on_result(result)
        return [result]

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "sweep", fake_sweep)
    response = CliRunner().invoke(cli.app, [command, "-s", "clip.mp4", "--device", "cpu", "--out", str(tmp_path)])
    assert response.exit_code == 0, response.output
    assert captured[0].precision == "fp32"
    assert captured[0].backend == "pytorch"


@pytest.mark.parametrize("settings", [
    {"precision": "int8"}, {"backend": "openvino"},
    {"device": "cpu", "precision": "fp16"},
    {"device": "mps", "precision": "amp-fp16"},
])
def test_api_rejects_unsupported_requests_before_session(monkeypatch, settings):
    create = Mock(side_effect=AssertionError("started"))
    monkeypatch.setattr(web.store, "create", create)
    response = TestClient(web.app).post("/api/sessions", json={"source": "clip.mp4", **settings})
    assert response.status_code == 422
    create.assert_not_called()


@pytest.mark.parametrize("precision", ["fp16", "amp-fp16"])
def test_api_rejects_auto_cpu_before_session(monkeypatch, precision):
    monkeypatch.setattr("visionbench.models.base.resolve_device", lambda _: "cpu")
    create = Mock(side_effect=AssertionError("started"))
    monkeypatch.setattr(web.store, "create", create)
    response = TestClient(web.app).post("/api/sessions", json={"source": "clip.mp4", "precision": precision})
    assert response.status_code == 422
    create.assert_not_called()


def test_api_forwards_execution_settings(monkeypatch):
    create = Mock(return_value=SimpleNamespace(status=lambda: {"id": "test"}))
    monkeypatch.setattr(web.store, "create", create)
    response = TestClient(web.app).post("/api/sessions", json={"source": "clip.mp4", "device": "cpu", "backend": "pytorch", "precision": "fp32"})
    assert response.status_code == 200
    config = create.call_args.args[0]
    assert config.backend == "pytorch"
    assert config.precision == "fp32"


def test_api_reports_unavailable_device_before_session(monkeypatch):
    def unavailable(_):
        raise ValueError("CUDA is unavailable")

    monkeypatch.setattr("visionbench.models.base.resolve_device", unavailable)
    create = Mock(side_effect=AssertionError("started"))
    monkeypatch.setattr(web.store, "create", create)
    response = TestClient(web.app).post("/api/sessions", json={"source": "clip.mp4", "device": "cuda"})
    assert response.status_code == 422
    assert "CUDA is unavailable" in response.json()["detail"]
    create.assert_not_called()
