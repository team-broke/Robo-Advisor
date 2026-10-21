"""원천 결함과 검증 상태를 보존하는 로컬 스냅샷 저장 계약을 확인한다."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import socket
import warnings

import numpy as np
import pandas as pd
import pytest

from robo_advisor.data.collectors import DatasetSpec, collect_fred, collect_yahoo
from robo_advisor.data import snapshot as snapshot_module
from robo_advisor.data.snapshot import (
    SnapshotChangedWarning,
    SnapshotError,
    SnapshotIntegrityError,
    SnapshotSchemaError,
    compare_manifests,
    compare_snapshots,
    dataframe_fingerprint,
    read_snapshot,
    write_snapshot,
)


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Snapshot tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def prices():
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [101.0, 102.0, 103.0],
            "Volume": [0, 20, 30],
        },
        index=pd.date_range("2019-09-02", periods=3, name="Date"),
    )


def _spec(**overrides):
    values = {
        "dataset_id": "SPY",
        "symbol": "SPY",
        "source": "yahoo",
        "role": "etf",
        "market": "US",
        "currency": "USD",
        "timezone": "America/New_York",
        "asset_class": "equity",
        "rationale": "Offline snapshot fixture",
        "selection_status": "candidate",
        "unit": "USD",
    }
    values.update(overrides)
    return DatasetSpec(**values)


class FakeResponse:
    text = "DATE,DGS3MO\r\n2019-09-02,2.0\r\n2019-09-03,0.0\r\n2019-09-04,-0.1\r\n"

    def raise_for_status(self):
        pass


@pytest.fixture
def results(prices):
    availability = pd.Series(
        pd.to_datetime(["2019-09-02T22:00Z", "2019-09-03T22:00Z", "2019-09-04T22:00Z"]),
        index=prices.index,
        name="available_at",
    )
    spy = collect_yahoo(
        _spec(),
        end="2019-09-05",
        download=lambda **kw: prices,
        trading_calendar=prices.index,
        available_at=availability,
    )
    fx_spec = _spec(
        dataset_id="USD_KRW",
        symbol="KRW=X",
        role="fx",
        market="FX",
        currency="KRW",
        timezone="Asia/Seoul",
        asset_class="currency",
        selection_status="confirmed",
        unit="KRW_per_USD",
    )
    fx = collect_yahoo(
        fx_spec,
        end="2019-09-05",
        download=lambda **kw: prices,
        trading_calendar=prices.index,
        quote_metadata={"symbol": "KRW=X", "base_currency": "USD", "quote_currency": "KRW"},
    )
    fred_spec = _spec(
        dataset_id="DGS3MO",
        symbol="DGS3MO",
        source="fred",
        role="risk_free",
        market="US_RATE",
        asset_class="interest_rate",
        unit="percent_per_annum",
    )
    fred = collect_fred(
        fred_spec,
        end="2019-09-05",
        http_get=lambda *args, **kw: FakeResponse(),
        trading_calendar=prices.index,
    )
    return {"SPY": spy, "USD_KRW": fx, "DGS3MO": fred}


def _manifest(path):
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))


def _entry(manifest, dataset_id="SPY"):
    return next(entry for entry in manifest["datasets"] if entry["dataset_id"] == dataset_id)


def _save_manifest(path, document):
    (path / "manifest.json").write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


def test_roundtrip_preserves_prices_rates_availability_and_validation(tmp_path, results):
    path = write_snapshot(results, root=tmp_path / "snapshots", snapshot_id="fixture-001")
    loaded = read_snapshot(path)
    assert loaded.path == path
    assert set(loaded.datasets) == set(results)
    assert loaded.manifest["schema_version"] == 1
    assert loaded.manifest["snapshot_id"] == "fixture-001"
    assert loaded.manifest["status"] == "complete"
    assert loaded.manifest["fingerprint_algorithm"] == "pandas-frame-v1"
    created = pd.Timestamp(loaded.manifest["snapshot_created_at"])
    assert created.tzinfo is not None and created.utcoffset().total_seconds() == 0
    for dataset_id, original in results.items():
        restored = loaded.datasets[dataset_id]
        pd.testing.assert_frame_equal(restored.data, original.data)
        pd.testing.assert_frame_equal(restored.raw_data, original.raw_data)
        assert restored.status == original.status == "unverified"
        assert restored.validation == original.validation
        assert restored.raw_text == original.raw_text
        safe_fields = (
            "dataset_id",
            "symbol",
            "source",
            "role",
            "market",
            "currency",
            "timezone",
            "collected_at",
            "requested_start",
            "requested_end_exclusive",
            "actual_start",
            "actual_end",
            "row_count",
            "price_basis",
            "auto_adjust",
            "availability_status",
        )
        assert {key: restored.metadata[key] for key in safe_fields} == {
            key: original.metadata[key] for key in safe_fields
        }
        original_at = original.metadata["available_at"]
        if original_at is None:
            assert restored.metadata["available_at"] is None
        else:
            pd.testing.assert_series_equal(restored.metadata["available_at"], original_at)
        entry = _entry(loaded.manifest, dataset_id)
        assert entry["ticker"] == original.metadata["symbol"]
        assert entry["status"] == original.status
        assert entry["row_count"] == len(original.data)
        assert entry["validation"] == original.validation
        assert entry["unverified_reasons"]
    assert list(loaded.datasets["DGS3MO"].data["Value"]) == [2.0, 0.0, -0.1]
    assert loaded.datasets["USD_KRW"].metadata["fx_direction"] == "USD/KRW"
    assert loaded.datasets["USD_KRW"].metadata["available_at"] is None


def test_manifest_file_hashes_and_fingerprints_match_saved_contents(tmp_path, results):
    path = write_snapshot(results, root=tmp_path)
    document = _manifest(path)
    for entry in document["datasets"]:
        for kind, item in entry["files"].items():
            if item is None:
                continue
            relative = Path(item["path"])
            assert not relative.is_absolute()
            target = path / relative
            assert target.resolve().is_relative_to(path.resolve())
            assert sha256(target.read_bytes()).hexdigest() == item["sha256"]
            if kind in {"data", "raw_data", "available_at"}:
                assert item["format"] == "parquet"
                assert len(item["fingerprint"]) == 64
    spy_entry = _entry(document)
    assert spy_entry["has_available_at"] is True
    assert spy_entry["files"]["available_at"] is not None
    assert spy_entry["files"]["raw_text"] is None
    assert _entry(document, "USD_KRW")["has_available_at"] is False


def test_auto_ids_create_independent_snapshots_and_explicit_ids_never_overwrite(tmp_path, results):
    first = write_snapshot(results, root=tmp_path)
    second = write_snapshot(results, root=tmp_path)
    assert first != second and first.parent == second.parent == tmp_path
    original_bytes = (first / "manifest.json").read_bytes()
    with pytest.raises(SnapshotError):
        write_snapshot(results, root=tmp_path, snapshot_id=first.name)
    assert (first / "manifest.json").read_bytes() == original_bytes
    assert read_snapshot(first).datasets["SPY"].status == "unverified"


@pytest.mark.parametrize(
    "dataset_id",
    ["..", ".", "../SPY", "SPY/data", "SPY\\data", "CON", "NUL", "COM1", "LPT9", "SPY "],
)
def test_unsafe_dataset_ids_are_rejected_before_publication(tmp_path, results, dataset_id):
    item = deepcopy(results["SPY"])
    item.metadata["dataset_id"] = dataset_id
    with pytest.raises(SnapshotSchemaError):
        write_snapshot({dataset_id: item}, root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_casefold_collisions_are_rejected(tmp_path, results):
    lower = deepcopy(results["SPY"])
    lower.metadata["dataset_id"] = "spy"
    with pytest.raises(SnapshotSchemaError):
        write_snapshot({"SPY": results["SPY"], "spy": lower}, root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_unsafe_snapshot_id_cannot_escape_root(tmp_path, results):
    root = tmp_path / "snapshots"
    with pytest.raises(SnapshotSchemaError):
        write_snapshot(results, root=root, snapshot_id="../escape")
    assert not (tmp_path / "escape").exists()


def test_original_data_and_caller_metadata_are_not_mutated(tmp_path, results):
    original = deepcopy(results)
    write_snapshot(results, root=tmp_path)
    for dataset_id, result in results.items():
        pd.testing.assert_frame_equal(result.data, original[dataset_id].data)
        pd.testing.assert_frame_equal(result.raw_data, original[dataset_id].raw_data)
        for key in ("status", "collected_at", "actual_start", "actual_end", "row_count", "issues"):
            assert result.metadata[key] == original[dataset_id].metadata[key]


def test_credentials_and_arbitrary_frame_attrs_are_not_written(tmp_path, results):
    item = deepcopy(results["SPY"])
    secret = "fixture-secret-credential-do-not-store"
    item.metadata["api_key"] = secret
    item.metadata["authorization"] = secret
    item.data.attrs["api_key"] = secret
    item.raw_data.attrs["api_key"] = secret
    path = write_snapshot({"SPY": item}, root=tmp_path)
    restored = read_snapshot(path).datasets["SPY"]
    assert "api_key" not in restored.metadata
    assert "authorization" not in restored.metadata
    assert restored.data.attrs == restored.raw_data.attrs == {}
    assert item.data.attrs["api_key"] == item.metadata["api_key"] == secret
    for target in path.rglob("*"):
        if target.is_file():
            assert secret.encode("utf-8") not in target.read_bytes()


def test_failed_source_is_persisted_with_duplicate_rows_and_original_findings(tmp_path, prices):
    source = prices.copy(deep=True)
    source.index = source.index[[0, 0, 2]]
    failed = collect_yahoo(_spec(), end="2019-09-05", download=lambda **kw: source)
    assert failed.status == "failed"
    path = write_snapshot({"SPY": failed}, root=tmp_path)
    restored = read_snapshot(path).datasets["SPY"]
    assert restored.status == "failed"
    assert restored.validation == failed.validation
    assert restored.data.index.has_duplicates
    pd.testing.assert_frame_equal(restored.raw_data, source)


@pytest.mark.parametrize("defect", ["nan", "infinity", "zero", "negative", "nat", "reversed"])
def test_snapshot_does_not_repair_or_drop_failed_raw_observations(tmp_path, prices, defect):
    source = prices.copy(deep=True)
    if defect in {"nan", "infinity", "zero", "negative"}:
        source.loc[source.index[1], "Close"] = {
            "nan": np.nan,
            "infinity": np.inf,
            "zero": 0.0,
            "negative": -1.0,
        }[defect]
    elif defect == "nat":
        source.index = pd.DatetimeIndex([source.index[0], pd.NaT, source.index[2]], name="Date")
    else:
        source.index = source.index[::-1]
    failed = collect_yahoo(_spec(), end="2019-09-05", download=lambda **kw: source)
    assert failed.status == "failed"
    path = write_snapshot({"SPY": failed}, root=tmp_path)
    restored = read_snapshot(path).datasets["SPY"]
    assert restored.status == "failed"
    assert len(restored.data) == len(restored.raw_data) == 3
    pd.testing.assert_frame_equal(restored.data, failed.data)
    pd.testing.assert_frame_equal(restored.raw_data, source)


def test_source_without_raw_frame_stays_absent_in_manifest_and_roundtrip(tmp_path):
    failed = collect_yahoo(_spec(), end="2019-09-05")
    assert failed.status == "failed" and failed.raw_data is None
    path = write_snapshot({"SPY": failed}, root=tmp_path)
    loaded = read_snapshot(path)
    assert _entry(loaded.manifest)["files"]["raw_data"] is None
    assert loaded.datasets["SPY"].raw_data is None
    assert loaded.datasets["SPY"].status == "failed"


def test_yfinance_multiindex_raw_columns_roundtrip_without_flattening(tmp_path, prices):
    source = prices.copy(deep=True)
    source.columns = pd.MultiIndex.from_tuples(
        [(name, "SPY") for name in prices.columns], names=["Price", "Ticker"]
    )
    result = collect_yahoo(_spec(), end="2019-09-05", download=lambda **kw: source)
    path = write_snapshot({"SPY": result}, root=tmp_path)
    restored = read_snapshot(path).datasets["SPY"]
    pd.testing.assert_frame_equal(restored.raw_data, source)
    assert isinstance(restored.raw_data.columns, pd.MultiIndex)
    assert not isinstance(restored.data.columns, pd.MultiIndex)


@pytest.mark.parametrize(
    "change", ["value", "row_order", "column_order", "dtype", "index_name", "timezone", "frequency"]
)
def test_dataframe_fingerprint_is_content_and_schema_sensitive(prices, change):
    original = dataframe_fingerprint(prices)
    changed = prices.copy(deep=True)
    if change == "value":
        changed.iloc[0, 0] += 0.5
    elif change == "row_order":
        changed = changed.iloc[::-1]
    elif change == "column_order":
        changed = changed.iloc[:, ::-1]
    elif change == "dtype":
        changed["Close"] = changed.Close.astype("float32")
    elif change == "index_name":
        changed.index.name = "session"
    elif change == "timezone":
        changed.index = changed.index.tz_localize("UTC")
    else:
        changed.index = pd.DatetimeIndex(list(changed.index), name="Date", freq=None)
    assert dataframe_fingerprint(prices.copy(deep=True)) == original
    assert dataframe_fingerprint(changed) != original


def test_dataframe_fingerprint_ignores_arbitrary_frame_attrs(prices):
    changed = prices.copy(deep=True)
    changed.attrs["ignored"] = "not a content or schema change"
    assert dataframe_fingerprint(changed) == dataframe_fingerprint(prices)


@pytest.mark.parametrize(
    "kind", ["mixed_object", "arbitrary_object", "complex", "category", "duplicate_columns"]
)
def test_unsupported_frame_types_fail_explicitly_without_a_partial_snapshot(
    tmp_path, results, kind
):
    item = deepcopy(results["SPY"])
    if kind == "mixed_object":
        item.data["Close"] = pd.Series([1, "2", None], index=item.data.index, dtype=object)
    elif kind == "arbitrary_object":
        item.data["Close"] = [object(), object(), object()]
    elif kind == "complex":
        item.data["Close"] = np.array([1 + 2j, 3 + 4j, 5 + 6j])
    elif kind == "category":
        item.data["Close"] = pd.Categorical([1, 2, 3])
    else:
        item.data = pd.concat([item.data, item.data[["Close"]]], axis=1)
    with pytest.raises(SnapshotSchemaError):
        write_snapshot({"SPY": item}, root=tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "dataset_id,kind",
    [
        ("SPY", "data"),
        ("SPY", "raw_data"),
        ("SPY", "available_at"),
        ("DGS3MO", "raw_text"),
    ],
)
def test_any_corrupted_payload_is_rejected_before_use(tmp_path, results, dataset_id, kind):
    path = write_snapshot(results, root=tmp_path)
    item = _entry(_manifest(path), dataset_id)["files"][kind]
    target = path / item["path"]
    target.write_bytes(target.read_bytes() + b"corrupted")
    with pytest.raises(SnapshotIntegrityError):
        read_snapshot(path)


def test_missing_payload_is_an_integrity_error(tmp_path, results):
    path = write_snapshot(results, root=tmp_path)
    item = _entry(_manifest(path))["files"]["data"]
    (path / item["path"]).unlink()
    with pytest.raises(SnapshotIntegrityError):
        read_snapshot(path)


def test_claimed_frame_fingerprint_is_verified_separately_from_byte_hash(tmp_path, results):
    path = write_snapshot(results, root=tmp_path)
    document = _manifest(path)
    _entry(document)["files"]["data"]["fingerprint"] = "0" * 64
    _save_manifest(path, document)
    with pytest.raises(SnapshotIntegrityError):
        read_snapshot(path)


@pytest.mark.parametrize(
    "unsafe_path", ["../outside.parquet", "../../outside.parquet", "C:/outside.parquet"]
)
def test_manifest_paths_cannot_escape_snapshot_directory(tmp_path, results, unsafe_path):
    path = write_snapshot(results, root=tmp_path)
    document = _manifest(path)
    _entry(document)["files"]["data"]["path"] = unsafe_path
    _save_manifest(path, document)
    with pytest.raises(SnapshotError):
        read_snapshot(path)


def test_payload_symlink_is_rejected_when_platform_can_create_one(tmp_path, results):
    path = write_snapshot(results, root=tmp_path / "snapshots")
    item = _entry(_manifest(path))["files"]["data"]
    target = path / item["path"]
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    try:
        target.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("This Windows session cannot create symbolic links")
    with pytest.raises(SnapshotError):
        read_snapshot(path)


@pytest.mark.parametrize(
    "problem", ["schema_version", "source_field", "created_at", "incomplete_status"]
)
def test_incomplete_or_incompatible_manifest_is_rejected(tmp_path, results, problem):
    path = write_snapshot(results, root=tmp_path)
    document = _manifest(path)
    if problem == "schema_version":
        document["schema_version"] = 999
    elif problem == "source_field":
        del _entry(document)["source"]
    elif problem == "created_at":
        del document["snapshot_created_at"]
    else:
        document["status"] = "incomplete"
    _save_manifest(path, document)
    with pytest.raises(SnapshotSchemaError):
        read_snapshot(path)


@pytest.mark.parametrize("problem", ["missing", "invalid_json"])
def test_manifest_is_required_and_must_be_valid_json(tmp_path, results, problem):
    path = write_snapshot(results, root=tmp_path)
    manifest = path / "manifest.json"
    if problem == "missing":
        manifest.unlink()
    else:
        manifest.write_text("{broken", encoding="utf-8")
    with pytest.raises(SnapshotError):
        read_snapshot(path)


def test_write_failure_cleans_only_its_own_partial_files(tmp_path, results, monkeypatch):
    previous = write_snapshot(results, root=tmp_path, snapshot_id="keep-snapshot")
    marker = tmp_path / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    before = {path.name for path in tmp_path.iterdir()}

    def fail_write(frame, path):
        raise OSError("injected write failure")

    monkeypatch.setattr(snapshot_module, "_write_frame", fail_write)
    with pytest.raises(SnapshotError):
        write_snapshot(results, root=tmp_path, snapshot_id="failed-snapshot")
    assert {path.name for path in tmp_path.iterdir()} == before
    assert marker.read_text(encoding="utf-8") == "keep"
    assert read_snapshot(previous).datasets["SPY"].status == "unverified"


def test_manifest_is_published_after_all_frame_writes(tmp_path, results, monkeypatch):
    original_write = snapshot_module._write_frame
    observed = []

    def observe(frame, path):
        assert not (tmp_path / "atomic-snapshot").exists()
        assert list(tmp_path.rglob("manifest.json")) == []
        observed.append(path)
        return original_write(frame, path)

    monkeypatch.setattr(snapshot_module, "_write_frame", observe)
    path = write_snapshot(results, root=tmp_path, snapshot_id="atomic-snapshot")
    assert observed
    assert read_snapshot(path).manifest["status"] == "complete"


def test_comparison_reports_added_removed_and_unchanged_dataset_ids(tmp_path, results):
    previous = write_snapshot({"SPY": results["SPY"], "USD_KRW": results["USD_KRW"]}, root=tmp_path)
    new = write_snapshot({"SPY": results["SPY"], "DGS3MO": results["DGS3MO"]}, root=tmp_path)
    difference = compare_snapshots(previous, new)
    assert difference["unchanged"] == ["SPY"]
    assert difference["added"] == ["DGS3MO"]
    assert difference["removed"] == ["USD_KRW"]
    assert difference["changed"] == []


def test_price_revision_is_detected_and_warns_when_previous_snapshot_is_supplied(tmp_path, results):
    previous = write_snapshot(results, root=tmp_path)
    revised = deepcopy(results)
    revised["SPY"].data.iloc[0, revised["SPY"].data.columns.get_loc("Close")] += 0.25
    with pytest.warns(SnapshotChangedWarning):
        new = write_snapshot(revised, root=tmp_path, previous=previous)
    difference = compare_snapshots(previous, new)
    assert len(difference["changed"]) == 1
    change = difference["changed"][0]
    assert change["dataset_id"] == "SPY"
    assert change["changes"]
    assert change["previous_fingerprint"] != change["new_fingerprint"]
    assert set(difference["unchanged"]) == {"USD_KRW", "DGS3MO"}
    assert difference["added"] == difference["removed"] == []


def test_identical_content_does_not_warn_merely_because_snapshot_id_and_time_changed(
    tmp_path, results
):
    previous = write_snapshot(results, root=tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        new = write_snapshot(results, root=tmp_path, previous=previous)
    assert not any(issubclass(item.category, SnapshotChangedWarning) for item in caught)
    difference = compare_snapshots(previous, new)
    assert set(difference["unchanged"]) == set(results)
    assert difference["changed"] == []


def _comparison_manifest(results, snapshot_id="comparison-fixture"):
    """A manifest fixture for pure comparison; it does not claim any disk payload exists."""
    entries = []
    for dataset_id, result in results.items():
        metadata = deepcopy(
            {key: value for key, value in result.metadata.items() if key != "available_at"}
        )
        files = {}
        for kind, frame in (
            ("data", result.data),
            ("raw_data", result.raw_data),
            (
                "available_at",
                (
                    result.metadata["available_at"].to_frame(name="available_at")
                    if result.metadata["available_at"] is not None
                    else None
                ),
            ),
        ):
            files[kind] = (
                None
                if frame is None
                else {
                    "path": f"{metadata['role']}/{dataset_id}/{kind}.parquet",
                    "format": "parquet",
                    "fingerprint_format": "pandas-frame-v1",
                    "sha256": "a" * 64,
                    "fingerprint": dataframe_fingerprint(frame),
                }
            )
        text = result.raw_text
        files["raw_text"] = (
            None
            if text is None
            else {
                "path": f"{metadata['role']}/{dataset_id}/raw_text.txt",
                "format": "utf-8",
                "sha256": sha256(text.encode()).hexdigest(),
                "fingerprint": sha256(text.encode()).hexdigest(),
            }
        )
        entry = {key: deepcopy(value) for key, value in metadata.items() if key != "issues"}
        entry.update(
            ticker=metadata["symbol"],
            metadata=metadata,
            validation=deepcopy(result.validation),
            validation_status=result.status,
            unverified_reasons=[
                item for item in metadata["issues"] if item["severity"] == "unverified"
            ],
            has_available_at=result.metadata["available_at"] is not None,
            files=files,
        )
        entries.append(entry)
    return {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "snapshot_created_at": "2026-10-15T00:00:00+00:00",
        "status": "complete",
        "fingerprint_algorithm": "pandas-frame-v1",
        "datasets": entries,
        "comparison": None,
    }


@pytest.mark.leakage
def test_pure_manifest_comparison_ignores_collection_time_and_parquet_bytes(results):
    old = _comparison_manifest(results)
    new = deepcopy(old)
    new["snapshot_id"] = "later-copy"
    new["snapshot_created_at"] = "2026-10-15T01:00:00+00:00"
    for entry in new["datasets"]:
        entry["collected_at"] = entry["metadata"]["collected_at"] = "2026-10-15T00:59:00Z"
        entry["files"]["data"]["sha256"] = "b" * 64
    difference = compare_manifests(old, new)
    assert set(difference["unchanged"]) == set(results)
    assert difference["changed"] == []
    assert _entry(new, "USD_KRW")["availability_status"] == "unverified"
    assert _entry(new, "USD_KRW")["has_available_at"] is False
    assert _entry(new, "USD_KRW")["files"]["available_at"] is None


@pytest.mark.parametrize("kind", ["data", "raw_data", "raw_text", "available_at"])
def test_pure_manifest_comparison_detects_each_logical_payload_change(results, kind):
    old = _comparison_manifest(results)
    new = deepcopy(old)
    dataset_id = "DGS3MO" if kind == "raw_text" else "SPY"
    _entry(new, dataset_id)["files"][kind]["fingerprint"] = "c" * 64
    difference = compare_manifests(old, new)
    assert difference["changed"][0]["dataset_id"] == dataset_id
    assert f"{kind}_fingerprint" in difference["changed"][0]["changes"]
    assert difference["changed"][0]["fingerprints"][kind]["previous"] != (
        difference["changed"][0]["fingerprints"][kind]["new"]
    )


@pytest.mark.parametrize(
    "key,value,group",
    [
        ("row_count", 4, "row_count"),
        ("actual_end", "2019-09-05", "date_range"),
        ("requested_end_exclusive", "2019-09-06", "date_range"),
        ("status", "failed", "status"),
        ("source", "revised-provider", "provider"),
        ("price_basis", "new adjustment definition", "price_definition"),
        ("auto_adjust", False, "price_definition"),
    ],
)
def test_pure_manifest_comparison_detects_quality_period_and_provider_changes(
    results, key, value, group
):
    old = _comparison_manifest(results)
    new = deepcopy(old)
    item = _entry(new)
    item[key] = item["metadata"][key] = value
    if key == "status":
        item["validation_status"] = value
    difference = compare_manifests(old, new)
    assert group in difference["changed"][0]["changes"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("fx_direction", "KRW/USD"),
        ("fx_direction_status", "unverified"),
        ("fx_quote_evidence", {"symbol": "other-quote"}),
    ],
)
def test_pure_manifest_comparison_includes_fx_contract_evidence(results, key, value):
    old = _comparison_manifest(results)
    new = deepcopy(old)
    _entry(new, "USD_KRW")["metadata"][key] = value
    difference = compare_manifests(old, new)
    assert difference["changed"][0]["dataset_id"] == "USD_KRW"
    assert "price_definition" in difference["changed"][0]["changes"]


def test_pure_manifest_comparison_rejects_secret_metadata(results):
    document = _comparison_manifest(results)
    _entry(document)["metadata"]["api_key"] = "fixture-not-a-real-key"
    with pytest.raises(SnapshotSchemaError):
        compare_manifests(document, deepcopy(document))


@pytest.mark.leakage
@pytest.mark.parametrize("operation", ["compare", "read"])
def test_verified_availability_requires_actual_payload(tmp_path, results, operation):
    path = write_snapshot(results, root=tmp_path) if operation == "read" else None
    document = _manifest(path) if path else _comparison_manifest(results)
    entry = _entry(document, "USD_KRW")
    assert entry["files"]["available_at"] is None
    entry["availability_status"] = entry["metadata"]["availability_status"] = "verified"
    if path:
        _save_manifest(path, document)
    with pytest.raises(SnapshotSchemaError, match="Verified availability"):
        if path:
            read_snapshot(path)
        else:
            compare_manifests(document, deepcopy(document))


@pytest.mark.parametrize("invalid", [[], None, "passed"])
def test_pure_manifest_comparison_rejects_invalid_validation_type(results, invalid):
    document = _comparison_manifest(results)
    _entry(document)["validation"] = invalid
    with pytest.raises(SnapshotSchemaError):
        compare_manifests(document, deepcopy(document))


def test_partial_payload_failure_removes_only_owned_staging(tmp_path, results, monkeypatch):
    marker = tmp_path / "keep.txt"
    marker.write_text("existing user file", encoding="utf-8")

    def write_partial(frame, path):
        path.write_bytes(b"partial")
        raise OSError("disk write interrupted")

    monkeypatch.setattr(snapshot_module, "_write_frame", write_partial)
    with pytest.raises(SnapshotError):
        write_snapshot(results, root=tmp_path, snapshot_id="interrupted")
    assert [path.name for path in tmp_path.iterdir()] == ["keep.txt"]
    assert marker.read_text(encoding="utf-8") == "existing user file"


def test_preexisting_empty_directory_is_never_replaced(tmp_path, results):
    existing = tmp_path / "already-present"
    existing.mkdir()
    with pytest.raises(SnapshotError):
        write_snapshot(results, root=tmp_path, snapshot_id=existing.name)
    assert existing.is_dir() and list(existing.iterdir()) == []


def test_preexisting_reservation_is_preserved(tmp_path, results):
    lock = tmp_path / ".snapshot-in-progress.lock"
    lock.write_text("another writer", encoding="utf-8")
    with pytest.raises(SnapshotError):
        write_snapshot(results, root=tmp_path, snapshot_id="in-progress")
    assert lock.read_text(encoding="utf-8") == "another writer"


@pytest.mark.parametrize("snapshot_id", ["", "NUL", "SPY."])
def test_additional_invalid_snapshot_names_fail_before_storage(tmp_path, results, snapshot_id):
    with pytest.raises(SnapshotSchemaError):
        write_snapshot(results, root=tmp_path, snapshot_id=snapshot_id)
    assert list(tmp_path.iterdir()) == []


def test_passed_benchmark_and_unnamed_availability_series_roundtrip(tmp_path, prices):
    availability = pd.Series(
        pd.to_datetime(["2019-09-02T22:00Z", "2019-09-03T22:00Z", "2019-09-04T22:00Z"]),
        index=prices.index,
    )
    result = collect_yahoo(
        _spec(
            dataset_id="SP500",
            symbol="^GSPC",
            role="benchmark",
            unit="index_points",
            selection_status="confirmed",
        ),
        end="2019-09-05",
        download=lambda **kwargs: prices,
        trading_calendar=prices.index,
        available_at=availability,
    )
    assert result.status == "passed"
    path = write_snapshot({"SP500": result}, root=tmp_path)
    restored = read_snapshot(path).datasets["SP500"]
    assert restored.status == "passed"
    assert _entry(_manifest(path), "SP500")["files"]["data"]["path"] == (
        "benchmark/SP500/data.parquet"
    )
    pd.testing.assert_series_equal(
        restored.metadata["available_at"], result.metadata["available_at"]
    )


@pytest.mark.parametrize("plain_object_columns", [False, True])
def test_empty_failed_data_preserves_original_columns_index_kind(tmp_path, plain_object_columns):
    result = collect_yahoo(_spec(), end="2019-09-05")
    if plain_object_columns:
        result.data = pd.DataFrame(columns=pd.Index([], dtype=object))
    path = write_snapshot({"SPY": result}, root=tmp_path)
    pd.testing.assert_frame_equal(read_snapshot(path).datasets["SPY"].data, result.data)
