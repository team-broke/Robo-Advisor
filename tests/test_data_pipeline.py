"""수집 → 스냅샷 실행 진입점을 네트워크 없이 확인한다."""

import json
import socket

import pandas as pd
import pytest

from robo_advisor.data import pipeline
from robo_advisor.data.snapshot import SnapshotChangedWarning, read_snapshot

CONFIG = "config/assets_team12.yaml"


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Pipeline tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def _prices(scale=1.0):
    index = pd.bdate_range("2019-09-02", periods=5, name="Date")
    close = [100.0 * scale + i for i in range(5)]
    return pd.DataFrame(
        {
            "Open": close,
            "High": [c + 1 for c in close],
            "Low": [c - 1 for c in close],
            "Close": close,
            "Volume": [10] * 5,
        },
        index=index,
    )


class FakeResponse:
    text = "DATE,DGS3MO\n2019-09-02,2.0\n2019-09-03,2.1\n2019-09-04,2.0\n"

    def raise_for_status(self):
        pass


def _run(tmp_path, scale=1.0, snapshot_id=None):
    return pipeline.run(
        CONFIG,
        end="2019-09-09",
        root=tmp_path,
        download=lambda **kw: _prices(scale),
        http_get=lambda *a, **kw: FakeResponse(),
        snapshot_id=snapshot_id,
    )


def test_collect_all_covers_every_configured_dataset():
    config = pipeline.load_asset_config(CONFIG)
    results = pipeline.collect_all(
        config,
        end="2019-09-09",
        download=lambda **kw: _prices(),
        http_get=lambda *a, **kw: FakeResponse(),
    )
    assert set(results) == {spec.dataset_id for spec in config["datasets"]}
    assert all(len(result.data) for result in results.values())


def test_without_network_permission_results_fail_but_are_preserved(tmp_path):
    report = pipeline.run(CONFIG, end="2019-09-09", root=tmp_path)
    assert report["counts"] == {"failed": len(report["datasets"])}
    assert {item["error"] for item in report["datasets"].values()} == {"NETWORK_NOT_AUTHORIZED"}
    snapshot = read_snapshot(report["snapshot"])
    assert len(snapshot.manifest["datasets"]) == len(report["datasets"])


def test_run_writes_manifest_and_links_previous_snapshot(tmp_path):
    first = _run(tmp_path, snapshot_id="20260101T000000000000Z_a")
    assert first["previous"] is None
    assert pipeline.latest_snapshot(tmp_path).name == "20260101T000000000000Z_a"

    second = _run(tmp_path, snapshot_id="20260102T000000000000Z_b")
    assert second["previous"] == first["snapshot"]
    comparison = read_snapshot(second["snapshot"]).manifest["comparison"]
    assert comparison["changed"] == [] and comparison["added"] == []


def test_changed_source_data_warns(tmp_path):
    _run(tmp_path, snapshot_id="20260101T000000000000Z_a")
    with pytest.warns(SnapshotChangedWarning):
        _run(tmp_path, scale=1.5, snapshot_id="20260102T000000000000Z_b")


def test_latest_snapshot_ignores_incomplete_directories(tmp_path):
    assert pipeline.latest_snapshot(tmp_path / "missing") is None
    (tmp_path / "20990101T000000000000Z_partial").mkdir()
    (tmp_path / ".staging-x").mkdir()
    assert pipeline.latest_snapshot(tmp_path) is None


def test_main_prints_report_and_signals_failures(tmp_path, capsys):
    code = pipeline.main(["--config", CONFIG, "--end", "2019-09-09", "--root", str(tmp_path)])
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["counts"]["failed"] == len(report["datasets"])


def test_main_rejects_invalid_config(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 2\n", encoding="utf-8")
    code = pipeline.main(["--config", str(bad), "--end", "2019-09-09", "--root", str(tmp_path)])
    assert code == 2
    assert "error:" in capsys.readouterr().err
