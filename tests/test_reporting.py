from pathlib import Path

import pytest

from visionbench import report


@pytest.mark.integration
def test_v2_save_load_and_report_preserve_execution_settings(tmp_path, result):
    path = report.save(result, tmp_path)
    saved = report.load_all(tmp_path)
    assert len(saved) == 1
    assert saved[0]["schema_version"] == 2
    assert saved[0]["_path"] == str(path)
    row = report.to_row(saved[0])
    assert row.backend == "pytorch"
    assert row.precision == "fp32"
    assert {"Backend", "Precision"} <= {column.header for column in report.rich_table(saved).columns}
    assert "pytorch" in report.markdown_table(saved)
    assert "fp32" in report.summarise(result)


def test_precision_and_backend_are_in_result_key(result):
    original = result.key
    result.model["precision"] = "amp-fp16"
    assert result.key != original
    assert "pytorch" in result.key
    assert "amp-fp16" in result.key


@pytest.mark.integration
def test_existing_v1_results_remain_readable():
    saved = [
        item for item in report.load_all(Path(__file__).resolve().parents[1] / "benchmarks")
        if item["schema_version"] == 1
    ]
    assert saved
    for item in saved:
        row = report.to_row(item)
        assert row.backend == "unknown"
        assert "unverified" in row.precision
    assert "unverified" in report.markdown_table(saved)


@pytest.mark.parametrize("data", [
    {"schema_version": 1},
    {"schema_version": 2, "config": {"precision": "fp16", "backend": "pytorch"}},
])
def test_missing_effective_settings_are_not_invented(data):
    row = report.to_row(data)
    assert row.backend == "unknown"
    assert row.precision == "unknown"


def test_v1_requested_fp16_is_not_reported_as_verified(result):
    data = result.to_dict()
    data["schema_version"] = 1
    data["model"]["precision"] = "fp16"
    row = report.to_row(data)
    assert row.precision == "fp16 (unverified)"
    assert row.backend == "unknown"
