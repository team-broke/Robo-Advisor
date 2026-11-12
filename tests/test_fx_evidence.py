"""Small synthetic archived responses; never replacements for actual FX proof."""

from hashlib import sha256
import json
import socket

import pandas as pd
import pytest

from robo_advisor.data.fx_evidence import audit_fx_evidence


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Evidence tests must not access provider APIs")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def _descriptor(path, root):
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }


def _fixture(root, source="alfred", *, mutation=None):
    folder = root / "evidence"
    folder.mkdir()
    if source == "alfred":
        original = b"observation_date,DEXKOUS_20240110\n2024-01-02,1300\n2024-01-03,.\n"
        name = "weekly/quote.csv"
        url = (
            "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=DEXKOUS&vintage_date=2024-01-10"
        )
    else:
        # Fixed literal renderer decoding, with a commented-out wrong value.
        original = (
            "<table><tr><td><script>d1('%_Z32%_Z30%_Z32%_Z34%_Z2e"
            "%_Z30%_Z31%_Z2e%_Z30%_Z32');</script>"
            "</td><td>USD</td><!-- <td>999</td> --><td>1,300.00</td></tr></table>"
        ).encode("cp949")
        name = "quote.html"
        url = "http://www.smbs.biz/ExRate/StdExRate.jsp?tongwha_code=USD"
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(original)
    digest = sha256(original).hexdigest()
    receipt = {
        "url": url,
        "http_status": 200,
        "sha256": digest,
        "bytes": len(original),
        "retrieved_at_utc": "2026-10-28T01:00:00+00:00",
        "actual_available_at": None,
    }
    if mutation == "receipt_hash":
        receipt["sha256"] = "0" * 64
    if mutation == "receipt_status":
        receipt["http_status"] = 403
    if mutation == "vintage_request":
        receipt["url"] = url.replace("2024-01-10", "2024-01-09")
    if mutation == "currency_request":
        receipt["url"] = url.replace("code=USD", "code=EUR")
    receipt_path = path.with_name(path.name + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    entries = [{**_descriptor(path, folder), "bytes": len(original)}]
    if mutation != "missing_receipt":
        entries.append({**_descriptor(receipt_path, folder), "bytes": receipt_path.stat().st_size})
    if source == "alfred":
        observed = pd.DataFrame(
            {
                "observation_date": ["2024-01-02", "2024-01-03"],
                "DEXKOUS": [1300.0, float("nan")],
                "vintage_date": ["2024-01-10"] * 2,
                "source_sha256": [digest] * 2,
                "source_file": [name] * 2,
                "actual_available_at": [None, None],
            }
        )
        if mutation == "source_hash":
            observed["source_sha256"] = "0" * 64
        if mutation == "future_vintage":
            observed["vintage_date"] = "2024-01-01"
        if mutation == "rate":
            observed.loc[0, "DEXKOUS"] = 1400.0
    else:
        observed = pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "mar_krw_per_usd": [1300.0],
                "evidence_sha": [digest],
                "actual_available_at": [None],
            }
        )
        if mutation == "rate":
            observed["mar_krw_per_usd"] = 1400.0
    csv = root / "observed.csv"
    observed.to_csv(csv, index=False)
    index = folder / "index.json"
    index.write_text(json.dumps({"training_approved": False, "files": entries}), encoding="utf-8")
    return {
        "source": source,
        "csv_descriptor": _descriptor(csv, root),
        "index_descriptor": _descriptor(index, root),
    }


@pytest.mark.parametrize("source, count, missing", [("smbs", 1, 0), ("alfred", 2, 1)])
def test_original_values_match_but_release_remains_unknown(tmp_path, source, count, missing):
    options = _fixture(tmp_path, source)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = audit_fx_evidence(tmp_path, **options)
    assert report["status"] == "unverified"
    assert report["training_approved"] is False
    assert report["metrics"]["original_values_status"] == "passed"
    assert report["metrics"]["matched_observations"] == count
    assert report["metrics"]["actual_release_missing"] == count
    assert report["metrics"]["null_observations"] == missing
    assert report["metrics"]["http_response_count"] == (1 if source == "smbs" else 0)
    assert before == {p: p.read_bytes() for p in before}
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "source,mutation",
    [
        ("alfred", "receipt_hash"),
        ("alfred", "receipt_status"),
        ("alfred", "missing_receipt"),
        ("alfred", "source_hash"),
        ("alfred", "vintage_request"),
        ("alfred", "rate"),
        ("smbs", "currency_request"),
        ("smbs", "rate"),
    ],
)
def test_response_chain_and_semantic_mismatches_are_failures(tmp_path, source, mutation):
    options = _fixture(tmp_path, source, mutation=mutation)
    report = audit_fx_evidence(tmp_path, **options)
    assert report["status"] == "failed"
    assert report["issues"][0]["code"] == "FX_EVIDENCE_INVALID"
    assert report["training_approved"] is False


@pytest.mark.leakage
def test_vintage_cannot_precede_its_observation(tmp_path):
    options = _fixture(tmp_path, mutation="future_vintage")
    report = audit_fx_evidence(tmp_path, **options)
    assert report["status"] == "failed"
    assert "after" in report["issues"][0]["message"]


@pytest.mark.parametrize("target", ["index", "payload", "csv"])
def test_changed_bytes_fail_external_anchors(tmp_path, target):
    options = _fixture(tmp_path)
    path = {
        "index": tmp_path / "evidence/index.json",
        "payload": tmp_path / "evidence/weekly/quote.csv",
        "csv": tmp_path / "observed.csv",
    }[target]
    with path.open("ab") as handle:
        handle.write(b"tamper")
    report = audit_fx_evidence(tmp_path, **options)
    assert report["status"] == "failed"


def test_path_escape_is_rejected_even_with_matching_content_hash(tmp_path):
    options = _fixture(tmp_path)
    index = tmp_path / "evidence/index.json"
    data = json.loads(index.read_text())
    data["files"][0]["path"] = "../observed.csv"
    index.write_text(json.dumps(data), encoding="utf-8")
    options["index_descriptor"] = _descriptor(index, tmp_path)
    assert audit_fx_evidence(tmp_path, **options)["status"] == "failed"


def test_unrelated_missing_response_is_recorded_without_inventing_it(tmp_path):
    options = _fixture(tmp_path)
    receipt = tmp_path / "evidence/unavailable.html.receipt.json"
    receipt.write_text(json.dumps({"http_status": 403}), encoding="utf-8")
    index = tmp_path / "evidence/index.json"
    data = json.loads(index.read_text())
    data["files"].append({**_descriptor(receipt, index.parent), "bytes": receipt.stat().st_size})
    index.write_text(json.dumps(data), encoding="utf-8")
    options["index_descriptor"] = _descriptor(index, tmp_path)
    out = audit_fx_evidence(tmp_path, **options)
    assert out["status"] == "unverified"
    assert out["metrics"]["index_integrity_status"] == "passed"
    assert out["metrics"]["preservation_status"] == "partial"
    assert out["metrics"]["original_values_status"] == "passed"
    assert out["metrics"]["unavailable_responses"] == [receipt.name]
    assert not (receipt.parent / "unavailable.html").exists()
