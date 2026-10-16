"""Offline legacy import: original cells and failed/unverified states survive."""

from __future__ import annotations

import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import socket
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

from robo_advisor.data.collectors import DatasetSpec, load_asset_config
from robo_advisor.data.importers import (
    LegacyImportError,
    import_legacy_csv,
    import_legacy_workbook,
    import_team12_sources,
)
from robo_advisor.data.snapshot import read_snapshot, write_snapshot

CSV_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Source",
    "Ticker",
]
WORKBOOK_COLUMNS = [
    "Date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Split numerator",
    "Split denominator",
    "Unit",
    "Source URL",
]
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Legacy import tests must never access external APIs")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def spy_spec():
    return DatasetSpec(
        dataset_id="SPY",
        symbol="SPY",
        source="yahoo",
        role="etf",
        market="US",
        currency="USD",
        timezone="America/New_York",
        asset_class="equity",
        rationale="Confirmed team universe; source quality remains unverified",
        selection_status="confirmed",
        unit="USD",
    )


@pytest.fixture
def ktb_spec():
    return DatasetSpec(
        dataset_id="KTB10Y",
        symbol="148070.KS",
        source="yahoo",
        role="etf",
        market="KRX",
        currency="KRW",
        timezone="Asia/Seoul",
        asset_class="bond",
        rationale="Preserve source error for source review",
        selection_status="confirmed",
        unit="KRW",
    )


def csv_row(day, *, ticker="SPY", **values):
    row = dict(
        zip(
            CSV_COLUMNS,
            [day, "101", "110", "99", "105", "87.5", "1200", "Yahoo Finance", ticker],
        )
    )
    row.update(values)
    return row


def write_csv(tmp_path, rows, *, columns=CSV_COLUMNS, name="source.csv"):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue().encode("utf-8")
    path = tmp_path / name
    path.write_bytes(payload)
    return path, payload


def workbook_row(day=43710, ticker="SPY", **values):
    # Excel epoch 1899-12-30: 43710 is 2019-09-02.
    row = dict(
        zip(
            WORKBOOK_COLUMNS,
            [
                day,
                ticker,
                "101",
                "110",
                "99",
                "105",
                "87.5",
                "1200",
                "0",
                "0",
                "0",
                "USD",
                f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            ],
        )
    )
    row.update(values)
    return row


def write_workbook(
    tmp_path,
    rows,
    *,
    shared=False,
    columns=WORKBOOK_COLUMNS,
    formula_cell=None,
    collected_serial="46281.36974476852",
    sheet_name="가격원자료",
):
    """Small valid OOXML package, without openpyxl or an Excel runtime."""
    strings = []

    def column_name(number):
        name = ""
        while number:
            number, remainder = divmod(number - 1, 26)
            name = chr(65 + remainder) + name
        return name

    def make_sheet(table):
        worksheet = ET.Element(f"{{{SHEET_NS}}}worksheet")
        data = ET.SubElement(worksheet, f"{{{SHEET_NS}}}sheetData")
        for row_number, values in enumerate(table, 1):
            row = ET.SubElement(data, f"{{{SHEET_NS}}}row", r=str(row_number))
            for column_number, value in enumerate(values, 1):
                if value is None:
                    continue
                address = f"{column_name(column_number)}{row_number}"
                cell = ET.SubElement(row, f"{{{SHEET_NS}}}c", r=address)
                if isinstance(value, (int, float)) or (address == "B3" and table[0][0] == "항목"):
                    ET.SubElement(cell, f"{{{SHEET_NS}}}v").text = str(value)
                elif shared:
                    cell.set("t", "s")
                    strings.append(str(value))
                    ET.SubElement(cell, f"{{{SHEET_NS}}}v").text = str(len(strings) - 1)
                else:
                    cell.set("t", "inlineStr")
                    inline = ET.SubElement(cell, f"{{{SHEET_NS}}}is")
                    ET.SubElement(inline, f"{{{SHEET_NS}}}t").text = str(value)
                if formula_cell == address and table[0][0] == "Date":
                    ET.SubElement(cell, f"{{{SHEET_NS}}}f").text = "1+100"
        return ET.tostring(worksheet, encoding="utf-8", xml_declaration=True)

    guide = make_sheet(
        [["항목", "내용"], ["데이터 범위", "Mock only"], ["수집 시각 UTC", collected_serial]]
    )
    prices = make_sheet([columns, *[[row.get(column) for column in columns] for row in rows]])
    workbook = ET.Element(f"{{{SHEET_NS}}}workbook")
    sheets = ET.SubElement(workbook, f"{{{SHEET_NS}}}sheets")
    for number, name in enumerate(["이용안내", sheet_name], 1):
        ET.SubElement(
            sheets,
            f"{{{SHEET_NS}}}sheet",
            name=name,
            sheetId=str(number),
            attrib={f"{{{REL_NS}}}id": f"rId{number}"},
        )
    relationships = ET.Element(f"{{{PACKAGE_REL_NS}}}Relationships")
    for number in [1, 2]:
        ET.SubElement(
            relationships,
            f"{{{PACKAGE_REL_NS}}}Relationship",
            Id=f"rId{number}",
            Type=f"{REL_NS}/worksheet",
            Target=f"worksheets/sheet{number}.xml",
        )
    path = tmp_path / "legacy.xlsx"
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", ET.tostring(workbook, encoding="utf-8"))
        archive.writestr("xl/_rels/workbook.xml.rels", ET.tostring(relationships, encoding="utf-8"))
        archive.writestr("xl/worksheets/sheet1.xml", guide)
        archive.writestr("xl/worksheets/sheet2.xml", prices)
        if shared:
            shared_strings = ET.Element(f"{{{SHEET_NS}}}sst")
            for value in strings:
                item = ET.SubElement(shared_strings, f"{{{SHEET_NS}}}si")
                ET.SubElement(item, f"{{{SHEET_NS}}}t").text = value
            archive.writestr("xl/sharedStrings.xml", ET.tostring(shared_strings, encoding="utf-8"))
    return path


def all_issues(result):
    issues = list(result.metadata["issues"])
    for report in result.validation.values():
        if isinstance(report, dict):
            issues.extend(report.get("issues", []))
    return issues


def test_csv_preserves_exact_bytes_raw_cells_and_adjustment_definition(tmp_path, spy_spec):
    path, payload = write_csv(tmp_path, [csv_row("2019-09-02"), csv_row("2019-09-03")])
    before = hashlib.sha256(payload).hexdigest()
    result = import_legacy_csv(spy_spec, path, end="2020-01-01", expected_sha256=before)
    assert path.read_bytes() == payload
    assert result.raw_text == payload.decode("utf-8")
    assert list(result.raw_data.columns) == CSV_COLUMNS
    assert result.raw_data.iloc[0]["Close"] == "105"
    assert result.raw_data.iloc[0]["Adj Close"] == "87.5"
    assert list(result.data.columns) == PRICE_COLUMNS
    assert result.data.iloc[0]["Close"] == 105
    assert result.data.iloc[0]["Adj Close"] == 87.5
    assert result.metadata["auto_adjust"] is None
    assert result.metadata["collected_at"] is None
    assert result.metadata["available_at"] is None
    assert result.metadata["availability_status"] == "unverified"
    assert result.status == "unverified"
    assert result.validation["price"]["status"] == "passed"
    assert any("ADJUSTMENT" in issue["code"] for issue in all_issues(result))


def test_range_is_exclusive_raw_source_retains_outside_rows(tmp_path, spy_spec):
    rows = [csv_row(day) for day in ["2019-08-30", "2019-09-02", "2020-01-01"]]
    path, _ = write_csv(tmp_path, rows)
    result = import_legacy_csv(spy_spec, path, end="2020-01-01")
    assert len(result.raw_data) == 3
    assert result.data.index.tolist() == [pd.Timestamp("2019-09-02")]
    assert result.metadata["requested_start"] == "2019-09-01"
    assert result.metadata["requested_end_exclusive"] == "2020-01-01"
    assert result.metadata["actual_start"] == "2019-09-02"
    assert result.metadata["actual_end"] == "2019-09-02"
    assert result.metadata["row_count"] == 1


def test_csv_invalid_dates_duplicates_and_reverse_rows_are_not_removed(tmp_path, spy_spec):
    rows = [csv_row(day) for day in ["2019-09-04", "not-a-date", "2019-09-02", "2019-09-02"]]
    path, _ = write_csv(tmp_path, rows)
    result = import_legacy_csv(spy_spec, path, end="2020-01-01")
    assert len(result.data) == 4
    assert result.data.index[0] == pd.Timestamp("2019-09-04")
    assert pd.isna(result.data.index[1])
    assert result.data.index.duplicated().sum() == 1
    assert result.raw_data["Date"].tolist() == [row["Date"] for row in rows]
    assert result.status == "failed"


@pytest.mark.parametrize("column,value", [("Close", ""), ("Open", "-1"), ("Low", "0")])
def test_actual_missing_and_nonpositive_prices_stay_failed(tmp_path, spy_spec, column, value):
    path, _ = write_csv(tmp_path, [csv_row("2019-09-02", **{column: value})])
    result = import_legacy_csv(spy_spec, path, end="2020-01-01")
    assert result.status == "failed"
    assert result.validation["price"]["status"] == "failed"
    assert result.raw_data.iloc[0][column] == value
    actual = result.data.iloc[0][column]
    assert pd.isna(actual) if value == "" else actual == float(value)


def test_non_numeric_cells_are_preserved_and_conversion_is_explicit_failure(tmp_path, spy_spec):
    path, _ = write_csv(tmp_path, [csv_row("2019-09-02", Close="not-a-price")])
    result = import_legacy_csv(spy_spec, path, end="2020-01-01")
    assert result.raw_data.iloc[0]["Close"] == "not-a-price"
    assert pd.isna(result.data.iloc[0]["Close"])
    assert result.status == "failed"
    assert any(issue["code"] == "SOURCE_NON_NUMERIC" for issue in all_issues(result))


def test_known_ktb_ohlc_error_is_not_fixed_or_downgraded(tmp_path, ktb_spec):
    row = csv_row("2020-03-10", ticker="148070.KS", Open="90", Low="99")
    path, payload = write_csv(tmp_path, [row])
    result = import_legacy_csv(ktb_spec, path, end="2021-01-01")
    assert result.status == "failed"
    assert result.validation["price"]["status"] == "passed"
    assert result.validation["ohlcv"]["status"] == "failed"
    assert any("OHLC" in issue["code"] for issue in all_issues(result))
    assert result.data.iloc[0]["Open"] == 90
    assert result.data.iloc[0]["Low"] == 99
    assert path.read_bytes() == payload


def test_domestic_date_difference_requires_calendar_evidence_not_filling(tmp_path, ktb_spec):
    rows = [csv_row(day, ticker="148070.KS") for day in ["2019-09-02", "2019-09-05"]]
    path, _ = write_csv(tmp_path, rows)
    result = import_legacy_csv(ktb_spec, path, end="2020-01-01")
    assert result.data.index.tolist() == list(pd.to_datetime(["2019-09-02", "2019-09-05"]))
    assert len(result.data) == 2
    assert result.status == "unverified"
    assert any("CALENDAR_UNVERIFIED" == issue["code"] for issue in all_issues(result))


def test_expected_hash_mismatch_rejects_source(tmp_path, spy_spec):
    path, payload = write_csv(tmp_path, [csv_row("2019-09-02")])
    with pytest.raises(LegacyImportError, match="(?i)sha|hash|fingerprint"):
        import_legacy_csv(spy_spec, path, end="2020-01-01", expected_sha256="0" * 64)
    assert path.read_bytes() == payload


@pytest.mark.parametrize("ticker", ["IWM", "KRW=X", "^KS11"])
def test_csv_ticker_mismatch_is_rejected(tmp_path, spy_spec, ticker):
    path, _ = write_csv(tmp_path, [csv_row("2019-09-02", ticker=ticker)])
    with pytest.raises(LegacyImportError, match="(?i)ticker|symbol"):
        import_legacy_csv(spy_spec, path, end="2020-01-01")


def test_normalized_v5_prices_are_rejected_as_source_ohlcv(tmp_path, spy_spec):
    path, _ = write_csv(
        tmp_path,
        [{"Date": "2020-01-02", "price_SPY": "100"}],
        columns=["Date", "price_SPY"],
    )
    with pytest.raises(LegacyImportError):
        import_legacy_csv(spy_spec, path, end="2021-01-01")


def test_duplicate_headers_are_not_silently_renamed(tmp_path, spy_spec):
    path = tmp_path / "duplicate.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Close,Adj Close,Volume,Source,Ticker\n"
        "2019-09-02,101,110,99,105,105,87.5,1200,Yahoo Finance,SPY\n",
        encoding="utf-8",
    )
    with pytest.raises(LegacyImportError):
        import_legacy_csv(spy_spec, path, end="2020-01-01")


@pytest.mark.parametrize("shared", [False, True])
def test_workbook_inline_and_shared_cells_preserve_original_fields(tmp_path, spy_spec, shared):
    rows = [workbook_row(43710), workbook_row(43711), workbook_row(43710, ticker="IWM")]
    path = write_workbook(tmp_path, rows, shared=shared)
    before = path.read_bytes()
    result = import_legacy_workbook(
        spy_spec, path, end="2020-01-01", expected_sha256=hashlib.sha256(before).hexdigest()
    )
    assert path.read_bytes() == before
    assert len(result.raw_data) == 2
    assert result.raw_data["Ticker"].tolist() == ["SPY", "SPY"]
    assert str(result.raw_data.iloc[0]["Date"]) == "43710"
    assert result.data.index.tolist() == list(pd.to_datetime(["2019-09-02", "2019-09-03"]))
    assert result.data.iloc[0]["Close"] == 105
    assert result.data.iloc[0]["Adj Close"] == 87.5
    assert result.metadata["auto_adjust"] is None
    assert result.status == "unverified"
    assert result.raw_text is None
    assert result.metadata["collected_at"] is not None
    assert result.metadata["available_at"] is None
    assert result.metadata["availability_status"] == "unverified"
    provenance = result.validation["legacy_import"]["metrics"]["provenance"]
    assert provenance["raw_schema"] == "xlsx-xml-lexical-cells-v1"
    assert provenance["sheet"] == "가격원자료"
    assert provenance["date_system"] == "1900"
    assert provenance["recorded_collection_time_verified"] is False
    assert result.raw_data["__excel_row"].tolist() == [2, 3]
    cells = json.loads(result.raw_data.iloc[0]["__excel_cells"])
    assert cells["A"]["ref"] == "A2"
    assert cells["A"]["type"] == "n"
    assert cells["A"]["xml_value"] == "43710"
    assert cells["C"]["ref"] == "C2"
    assert cells["C"]["type"] == ("s" if shared else "inlineStr")


def test_workbook_missing_cell_stays_missing_not_forward_filled(tmp_path, spy_spec):
    path = write_workbook(tmp_path, [workbook_row(43711, Close=None), workbook_row(43712)])
    result = import_legacy_workbook(spy_spec, path, end="2020-01-01")
    assert result.raw_data.iloc[0]["Close"] is None
    assert pd.isna(result.data.iloc[0]["Close"])
    assert result.data.iloc[1]["Close"] == 105
    assert result.status == "failed"


def test_workbook_unit_mismatch_stays_failed_without_currency_conversion(tmp_path, spy_spec):
    path = write_workbook(tmp_path, [workbook_row(43710, Unit="KRW")])
    original_bytes = path.read_bytes()
    result = import_legacy_workbook(spy_spec, path, end="2020-01-01")
    assert result.status == "failed"
    assert any(issue["code"] == "SOURCE_UNIT_MISMATCH" for issue in all_issues(result))
    assert result.raw_data.iloc[0]["Unit"] == "KRW"
    assert result.metadata["currency"] == "USD"
    assert result.data.iloc[0]["Close"] == 105
    assert path.read_bytes() == original_bytes


def test_workbook_formula_cells_are_not_treated_as_verified_cached_prices(tmp_path, spy_spec):
    path = write_workbook(tmp_path, [workbook_row(43711)], formula_cell="C2")
    with pytest.raises(LegacyImportError, match="(?i)formula"):
        import_legacy_workbook(spy_spec, path, end="2020-01-01")


def test_workbook_absent_ticker_is_not_substituted(tmp_path, spy_spec):
    path = write_workbook(tmp_path, [workbook_row(43711, ticker="IWM")])
    with pytest.raises(LegacyImportError, match="(?i)ticker|symbol|SPY"):
        import_legacy_workbook(spy_spec, path, end="2020-01-01")


def test_workbook_wrong_schema_is_rejected(tmp_path, spy_spec):
    path = write_workbook(tmp_path, [workbook_row(43711)], sheet_name="normalized_v5")
    with pytest.raises(LegacyImportError):
        import_legacy_workbook(spy_spec, path, end="2020-01-01")


@pytest.mark.parametrize(
    "defect", ["negative-shared-index", "invalid-row-number", "cell-row-mismatch"]
)
def test_workbook_rejects_ambiguous_structural_cell_references(tmp_path, spy_spec, defect):
    path = write_workbook(tmp_path, [workbook_row(43710)], shared=True)
    with ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    target = "xl/worksheets/sheet2.xml"
    worksheet = ET.fromstring(members[target])
    rows = worksheet.find(f"{{{SHEET_NS}}}sheetData")
    if defect == "negative-shared-index":
        rows[0][0].find(f"{{{SHEET_NS}}}v").text = "-1"
    elif defect == "invalid-row-number":
        rows[1].set("r", "invalid")
    else:
        rows[1][2].set("r", "C3")  # the cell belongs to enclosing row 2
    members[target] = ET.tostring(worksheet, encoding="utf-8")
    with ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    corrupted_bytes = path.read_bytes()
    with pytest.raises(LegacyImportError):
        import_legacy_workbook(spy_spec, path, end="2020-01-01")
    assert path.read_bytes() == corrupted_bytes


@pytest.mark.parametrize("bad_serial", ["invalid", "", None])
def test_workbook_bad_collection_timestamp_never_creates_fake_available_at(
    tmp_path, spy_spec, bad_serial
):
    path = write_workbook(tmp_path, [workbook_row(43711)], collected_serial=bad_serial)
    result = import_legacy_workbook(spy_spec, path, end="2020-01-01")
    assert result.metadata["collected_at"] is None
    assert result.metadata["available_at"] is None
    assert result.status != "passed"


@pytest.mark.leakage
def test_file_mtime_is_never_historical_availability(tmp_path, spy_spec):
    path, _ = write_csv(tmp_path, [csv_row("2019-09-02")])
    result = import_legacy_csv(spy_spec, path, end="2020-01-01")
    assert result.metadata["available_at"] is None
    assert result.metadata["collected_at"] is None
    assert result.metadata["availability_status"] == "unverified"
    assert any("AVAILABILITY" in issue["code"] for issue in all_issues(result))


@pytest.mark.parametrize("failed", [False, True])
def test_actual_parquet_roundtrip_preserves_status_raw_anomalies_and_source(
    tmp_path, spy_spec, failed
):
    rows = [csv_row("2019-09-02"), csv_row("2019-09-03", Close="" if failed else "105")]
    path, original_bytes = write_csv(tmp_path, rows)
    result = import_legacy_csv(spy_spec, path, end="2020-01-01")
    assert result.status == ("failed" if failed else "unverified")
    saved = write_snapshot(
        {"SPY": result}, root=tmp_path / "snapshots", snapshot_id="legacy-source"
    )
    restored = read_snapshot(saved).datasets["SPY"]
    assert restored.status == result.status
    assert restored.metadata["available_at"] is None
    assert_frame_equal(restored.data, result.data)
    assert_frame_equal(restored.raw_data, result.raw_data)
    assert restored.raw_text == result.raw_text
    assert path.read_bytes() == original_bytes


def test_actual_parquet_keeps_clean_source_unverified(tmp_path, spy_spec):
    path, _ = write_csv(tmp_path, [csv_row("2019-09-02"), csv_row("2019-09-03")])
    result = import_legacy_csv(spy_spec, path, end="2020-01-01")
    saved = write_snapshot({"SPY": result}, root=tmp_path / "snapshots", snapshot_id="unverified")
    snapshot = read_snapshot(saved)
    assert snapshot.datasets["SPY"].status == "unverified"
    assert snapshot.manifest["datasets"][0]["validation_status"] == "unverified"


def test_actual_parquet_retains_bad_dates_duplicate_observations_and_missing_price(
    tmp_path, spy_spec
):
    rows = [
        csv_row("2019-09-03"),
        csv_row("invalid"),
        csv_row("2019-09-02", Close=""),
        csv_row("2019-09-02"),
    ]
    path, payload = write_csv(tmp_path, rows)
    result = import_legacy_csv(spy_spec, path, end="2020-01-01")
    saved = write_snapshot({"SPY": result}, root=tmp_path / "snapshots", snapshot_id="bad-source")
    restored = read_snapshot(saved).datasets["SPY"]
    assert restored.status == "failed"
    assert len(restored.data) == 4
    assert pd.isna(restored.data.index[1])
    assert restored.data.index.duplicated().sum() == 1
    assert pd.isna(restored.data.iloc[2]["Close"])
    assert_frame_equal(restored.data, result.data)
    assert_frame_equal(restored.raw_data, result.raw_data)
    assert path.read_bytes() == payload


def make_team12_batch(
    tmp_path, *, peer_ticker="^KS11", peer_source="Yahoo Finance", include_reference_hash=True
):
    workbook_assets = ["SPY", "IWM", "EFA", "EEM", "IEF", "TIP", "LQD", "GLD", "VNQ"]
    rows = [workbook_row(43710, ticker=ticker) for ticker in workbook_assets]
    workbook = write_workbook(tmp_path, rows)
    csv_root = tmp_path / "raw"
    csv_root.mkdir()
    originals = {workbook: workbook.read_bytes()}
    for dataset_id, ticker in [
        ("DBC", "DBC"),
        ("KODEX_200TR", "278530.KS"),
        ("KTB10Y", "148070.KS"),
    ]:
        row = csv_row("2019-09-02", ticker=ticker)
        if dataset_id == "KTB10Y":
            row["Open"] = "90"  # actual source error remains failed
        path, payload = write_csv(csv_root, [row], name=f"{dataset_id}_2019_2025.csv")
        originals[path] = payload
    reference, reference_bytes = write_csv(
        csv_root,
        [
            csv_row("2019-09-02", ticker=peer_ticker, Source=peer_source),
            csv_row("2019-09-03", ticker=peer_ticker, Source=peer_source),
        ],
        name="KOSPI_2019_2025.csv",
    )
    originals[reference] = reference_bytes
    original_config = Path(__file__).resolve().parents[1] / "config" / "assets_team12.yaml"
    original_config_bytes = original_config.read_bytes()
    config = yaml.safe_load(original_config_bytes)
    for source in config["legacy_import"]["sources"].values():
        source_path = workbook if source["kind"] == "workbook" else csv_root / source["filename"]
        source["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    config["legacy_import"]["reference_kospi"]["sha256"] = hashlib.sha256(
        reference_bytes
    ).hexdigest()
    if not include_reference_hash:
        del config["legacy_import"]["reference_kospi"]["sha256"]
    config_path = tmp_path / "assets_team12.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return {
        "workbook_assets": workbook_assets,
        "workbook": workbook,
        "csv_root": csv_root,
        "config_path": config_path,
        "originals": originals,
        "original_config": original_config,
        "original_config_bytes": original_config_bytes,
    }


def test_team12_batch_import_uses_only_selected_original_sources(tmp_path):
    batch = make_team12_batch(tmp_path)
    imported = import_team12_sources(
        batch["config_path"],
        workbook_path=batch["workbook"],
        csv_root=batch["csv_root"],
        end="2020-01-01",
    )
    assert set(imported) == set(batch["workbook_assets"] + ["DBC", "KODEX_200TR", "KTB10Y"])
    assert imported["KTB10Y"].status == "failed"
    assert all(result.status == "unverified" for key, result in imported.items() if key != "KTB10Y")
    assert all(result.metadata["available_at"] is None for result in imported.values())
    for dataset_id in ["KODEX_200TR", "KTB10Y"]:
        report = imported[dataset_id].validation["peer_observations"]
        assert report["status"] == "unverified"
        assert report["metrics"]["official_calendar"] is False
        assert report["issues"][0]["code"] == "OBSERVED_DATE_DIFFERENCE"
        assert report["issues"][0]["count"] == 1
        assert report["issues"][0]["dates"] == ["2019-09-03"]
        assert imported[dataset_id].data.index.tolist() == [pd.Timestamp("2019-09-02")]
    assert all(path.read_bytes() == payload for path, payload in batch["originals"].items())
    assert batch["original_config"].read_bytes() == batch["original_config_bytes"]


@pytest.mark.parametrize(
    "peer_settings",
    [
        {"peer_ticker": "SPY"},
        {"peer_source": "FRED"},
        {"include_reference_hash": False},
    ],
    ids=["wrong-ticker", "wrong-source", "missing-audited-hash"],
)
def test_batch_rejects_unidentified_or_unpinned_peer_reference(tmp_path, peer_settings):
    batch = make_team12_batch(tmp_path, **peer_settings)
    with pytest.raises(LegacyImportError):
        import_team12_sources(
            batch["config_path"],
            workbook_path=batch["workbook"],
            csv_root=batch["csv_root"],
            end="2020-01-01",
        )
    assert all(path.read_bytes() == payload for path, payload in batch["originals"].items())
    assert batch["original_config"].read_bytes() == batch["original_config_bytes"]


def test_team12_configuration_has_exact_confirmed_assets_without_rf_assumption():
    path = Path(__file__).resolve().parents[1] / "config" / "assets_team12.yaml"
    config = load_asset_config(path)
    expected = [
        "SPY",
        "IWM",
        "EFA",
        "EEM",
        "IEF",
        "TIP",
        "LQD",
        "GLD",
        "DBC",
        "VNQ",
        "KODEX_200TR",
        "KTB10Y",
    ]
    assert config["universe"]["status"] == "confirmed"
    assert config["universe"]["selected_etfs"] == expected
    assert config["risk_free_policy"]["status"] == "unconfirmed"
    etfs = [spec for spec in config["datasets"] if spec.role == "etf"]
    assert [spec.dataset_id for spec in etfs] == expected
    assert all(spec.selection_status == "confirmed" for spec in etfs)
    assert not {"TLT", "QQQ", "AGG", "KODEX_200"}.intersection({spec.dataset_id for spec in etfs})
