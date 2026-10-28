# 원천 데이터 스냅샷 (#11)

> 현재 적용 상태(2026-10-28): #11 스냅샷 모듈은 develop에 반영되어 있으며,
> PR #106 감사기가 저장 파일과 원천 상태를 구분해 검증한다. 아래 구현 당시의
> 실행 기록은 당시 환경에 관한 기록이다. 현재 원천 증빙과 미확인 항목은
> [원천 증거 검증](source-evidence.md)을 참조한다.

## 목적과 범위

스냅샷은 #9 수집기의 `CollectionResult`를 저장하고 다시 읽을 때 파일과 논리 데이터의
일관성을 확인하는 계층이다. 수집 원본, 정규화 데이터, 검증 결과와 미검증 사유를
함께 보존한다. #9 수집기와 #10 검증기는 수정하지 않으며 실험 유니버스의 #84 코드가
없어도 사용할 수 있다.

외부 API 호출·재수집은 수행하지 않는다. 기존 v5 파일을 읽거나 실제 15개 ETF와
보조 자료 4개의 스냅샷을 생성하지 않는다. 최종 ETF 12종과 Rf 정책도 확정하지 않는다.
기본 저장 위치는 `data/raw/snapshots`이며 오프라인 예제와 테스트는 임시 위치를 사용한다.

## 공개 API

```text
write_snapshot(
    results: dict[str, CollectionResult],
    root="data/raw/snapshots",
    *,
    snapshot_id=None,
    previous=None,
) -> Path

read_snapshot(path, *, expected_manifest_sha256=None) -> Snapshot

compare_snapshots(previous, new) -> dict

compare_manifests(previous: dict, new: dict) -> dict

dataframe_fingerprint(frame) -> str
```

`write_snapshot`은 dataset_id를 키로 받으며 완성된 스냅샷 디렉터리의 `Path`를 반환한다.
`previous`와 `compare_snapshots`의 입력은 스냅샷 경로 또는 검증된 `Snapshot`이다.
파일 무결성 확인에는 manifest dict를 대신 전달하지 않는다.
`read_snapshot`의 반환 객체는 `path`, `manifest`, `datasets`, `manifest_sha256`를 갖는다.
`Snapshot.datasets`는 dataset_id별로 복원한 `CollectionResult`의 dict다.
`compare_manifests`는 manifest 두 개의 구조와 내용을 비교하는 순수 함수다.
파일을 읽거나 무결성을 검증하지 않으므로 실제 저장본 비교에는
`compare_snapshots`를 사용한다. 이 함수는 경로/객체의 저장본을 먼저 검증한 뒤
manifest 비교를 수행한다.

## 저장 구조와 원본 보존

데이터셋의 `role`을 따라 ETF, 벤치마크, FX, Rf 자료를 구분한다. 각 실제 파일의
상대 경로는 manifest의 `files` descriptor가 기록한다. 소비자가 디렉터리 이름이나
파일명에서 티커·통화·역할을 추측하지 않는다.
예를 들어 SPY의 정규화 자료는 스냅샷 아래 `etf/SPY/data.parquet`에 기록하고
같은 데이터셋의 원본·원문·가용성 파일은 별도로 둔다.

| 파일 종류 | 내용 |
|---|---|
| `data` | 정규화 DataFrame의 typed Parquet |
| `raw_data` | 원본 DataFrame의 별도 typed Parquet; 원본이 없으면 `None` |
| `raw_text` | FRED 등에서 확보한 원문을 보존하는 별도 UTF-8 텍스트; 없으면 `None` |
| `available_at` | 실제로 제공된 역사적 시각 Series의 별도 Parquet; 없으면 `None` |

잘못된 값, 결측, 중복/역순 날짜, 0 또는 음수 값을 삭제·정렬·보간하지 않는다.
이상 값이 있는 자료도 지원하는 스키마이면 그대로 저장한다. 저장 계층은 가격 검증을
재실행하거나 #9/#10이 반환한 상태를 자동 승인하지 않는다.

DataFrame은 native typed Parquet로 기록하고 손실 없는 왕복 복원을 확인한다.
지원하지 않는 타입이나 구조를 문자열로 바꾸거나 pickle로 저장하지 않는다.
혼합 타입 object 열, 임의 Python 객체, category, period, complex, 중복 열 등의
지원 불가 스키마에는 `SnapshotSchemaError`를 발생시킨다. 원천 상태가 `failed`라는 이유로
저장을 거부하는 것과 저장 형식으로 보존할 수 없는 스키마를 거부하는 것은 구분한다.
DataFrame `attrs`는 논리 데이터 계약에 포함하지 않는다.

`available_at`을 받은 경우 그 값·인덱스·시간대를 별도 파일로 보존하고 읽을 때
`CollectionResult.metadata["available_at"]` Series로 복원한다. Series 이름은
`None` 또는 문자열을 지원한다. 원천에 시각 근거가 없으면 `None`을 유지한다.
스냅샷 작성 시각을 과거 공개시각으로 대신 넣지 않는다.
읽을 때도 `availability_status="verified"`에 실제 `available_at` 파일이 없으면
manifest 오류로 거부한다.

## Manifest 계약

최상위 manifest는 아래 정보를 갖는다.

| 필드 | 의미 |
|---|---|
| `schema_version` | 저장/복원 manifest 형식 버전 |
| `snapshot_id` | 덮어쓸 수 없는 스냅샷 식별자 |
| `snapshot_created_at` | 이번 스냅샷 작성 시각, UTC |
| `status` | `complete`: 저장이 끝났다는 뜻 |
| `fingerprint_algorithm` | `pandas-frame-v1` |
| `datasets` | 데이터셋별 manifest 항목의 list |
| `comparison` | 이전 스냅샷과의 비교 결과 또는 `None` |

`status="complete"`는 데이터 품질 검증이나 학습 사용 승인과 별개다.
데이터셋의 `passed`/`failed`/`unverified`는 각 항목 안에 원래대로 남긴다.

데이터셋 항목에는 다음 필드를 기록한다.

| 필드 | 의미 |
|---|---|
| `dataset_id`, `ticker`, `source`, `role`, `market`, `currency` | 수집 명세와 원천 식별 |
| `collected_at` | #9 원천 수집 UTC 시각; 원천에 없으면 `None` |
| `requested_start`, `requested_end_exclusive` | 요청 기간의 시작일과 배타적 종료일 |
| `actual_start`, `actual_end`, `row_count` | 원천이 기록한 실제 기간과 행 수 |
| `price_basis`, `auto_adjust` | 가격 정의와 Yahoo 조정 옵션 |
| `status`, `validation_status` | 원래 `CollectionResult.status` |
| `validation` | #10 및 수집 어댑터 검증 보고서 |
| `unverified_reasons` | 미검증 근거 |
| `availability_status`, `has_available_at` | 가용성 상태와 실제 Series 존재 여부 |
| `files` | `data`, `raw_data`, `raw_text`, `available_at`의 descriptor 또는 `None` |

스냅샷 시각과 원천 수집시각을 구분한다. 작성 시점이 바뀌어도 과거 `collected_at`을
현재 시각으로 갱신하지 않는다. 원천의 실제 기간·행 수와 저장 데이터가 일치하는지도
확인하며, 해석할 수 없는 날짜의 기간을 임의로 추정해 추가하지 않는다.

DataFrame descriptor는 `path`, `sha256`, `fingerprint`를 구분해 기록한다.
`raw_text`는 원문 파일의 해시로 추적한다. 다음은 의미를 보여주는 일부 필드 예시다.
경로와 해시는 실제 작성 결과의 manifest를 사용한다.

```json
{
  "schema_version": 1,
  "snapshot_id": "demo-a",
  "snapshot_created_at": "<스냅샷 작성 UTC 시각>",
  "status": "complete",
  "fingerprint_algorithm": "pandas-frame-v1",
  "datasets": [
    {
      "dataset_id": "SPY",
      "ticker": "SPY",
      "source": "yahoo",
      "role": "etf",
      "collected_at": null,
      "status": "unverified",
      "validation_status": "unverified",
      "availability_status": "unverified",
      "has_available_at": false,
      "files": {
        "data": {
          "path": "etf/SPY/data.parquet",
          "format": "parquet",
          "sha256": "<파일 바이트의 SHA-256>",
          "fingerprint": "<DataFrame 논리 내용의 SHA-256>"
        },
        "available_at": null
      }
    }
  ],
  "comparison": null
}
```

Manifest의 메타데이터는 #9의 안전한 필드 allowlist를 사용한다. 임의의 추가
adapter metadata 전체를 보존하는 계약이 아니다. credential/token/인증정보 키는
재귀적으로 제거 또는 가림 처리하며 원천 DataFrame과 가용성 Series를 metadata JSON의
문자열 표현으로 바꾸지 않는다. 비밀 값이 포함된 객체의 `repr`를 저장하지 않는다.

## 파일 해시와 논리 fingerprint

`sha256`은 저장된 파일 바이트의 무결성 검사다. Parquet 직렬화 방식이 달라지면
논리 데이터가 같아도 파일 바이트 해시는 달라질 수 있다.
`dataframe_fingerprint`는 `pandas-frame-v1` 규칙으로 값·열·인덱스·순서·dtype·결측을
포함한 논리 내용을 결정적으로 계산한다. 인덱스나 행 순서가 바뀌면 다른 데이터다.
DataFrame `attrs`는 제외한다.

단순히 CSV 문자열로 바꾸거나 열/행을 정렬해서 해시하지 않는다. 문자열 숫자와 숫자,
시간대 및 dtype 차이를 없애면서 같다고 판정하지 않는다. `data`, `raw_data`,
실제 `available_at` 각각의 내용 변경을 구분한다.

`read_snapshot`은 manifest와 파일 경로/스키마를 검사하고 각 파일의 SHA-256 및
복원한 DataFrame fingerprint를 확인한다. 불일치하면 결과를 반환하지 않고 오류를
발생시킨다. 절대경로, 상위 경로 이동 또는 스냅샷 밖으로 나가는 파일 참조를 허용하지 않는다.
손상되거나 누락된 데이터를 빈 DataFrame으로 대체하지 않는다.

이 검사는 manifest와 파일의 내부 일관성을 검증한다. 원천 제공자의 진위나 역사적
공개시각을 새로 인증하는 기능은 아니다. #15 소비자는 검증된 로더를 통해 자료를 읽는다.
이미 신뢰할 수 있는 별도 계보에 manifest SHA-256을 기록했다면
`expected_manifest_sha256`으로 전달해 manifest 자체의 변경도 확인할 수 있다.
파일과 manifest를 함께 임의로 다시 작성한 공격까지 기본 내부 해시 검사만으로 인증하지는
않는다. 저장 API의 불변성과 운영체제 차원의 WORM 저장소는 구분한다.

## 원자적 작성과 덮어쓰기 금지

작성은 대상과 같은 파일시스템의 임시 staging 디렉터리에서 수행한다. 동일 ID에
대한 exclusive reservation으로 동시 작성 충돌을 막고, 파일·manifest·왕복 복원 검사까지
성공한 파일을 독점 생성한 최종 디렉터리로 옮기고, `manifest.json`을 마지막에
원자적으로 공개한다. 디렉터리 전체를 한 번에 rename하는 방식은 아니다.
최종 디렉터리가 먼저 보이더라도 manifest가 없으면 로더는 미완료로 거부한다.

이미 존재하는 snapshot_id는 실패하며 기존 파일을 덮어쓰지 않는다. 이전 스냅샷을
최신 데이터로 갱신하는 저장 방식도 사용하지 않는다. 실패한 임시 작성 결과를
완성된 스냅샷으로 읽지 않으며, 공개된 이전 스냅샷의 내용과 상태는 유지한다.
입력 스키마·직렬화·I/O 오류가 발생하면 완성된 저장으로 보고하지 않는다.
호출 중 발생한 오류는 해당 호출이 독점 생성한 staging/최종 디렉터리만 정리한다.
프로세스 강제 종료로 남은 lock/미완료 디렉터리는 자동으로 기존 스냅샷처럼 읽거나
덮어쓰지 않으며 운영자가 확인해야 한다.

## 이전 스냅샷 비교

`write_snapshot(..., previous=...)`는 이전 자료와의 비교를 manifest에 남긴다.
`compare_snapshots(previous, new)`는 아래 형태로 분류한다.

```text
{
    "unchanged": [dataset_id, ...],
    "added": [dataset_id, ...],
    "removed": [dataset_id, ...],
    "changed": [
        {
            "dataset_id": ...,
            "changes": [...],
            "previous_fingerprint": ...,
            "new_fingerprint": ...,
            "fingerprints": {kind: {"previous": ..., "new": ...}, ...},
        },
        ...,
    ],
}
```

값뿐 아니라 원본·공개시각·원문, 스키마, 상태, 기간, 공급자 및 가격 정의 등의
의미 있는 변경을 비교한다. `collected_at`, snapshot ID 및 스냅샷 작성 시각처럼
이번 실행에 따라 달라지는 값만 바뀌었다면 논리 변경으로 취급하지 않는다.
물리 Parquet 해시만 다른 것을 곧바로 원천 내용 변경으로 판정하지 않는다.
`previous_fingerprint`/`new_fingerprint`는 정규화 `data`의 fingerprint다.
원본만 바뀌면 이 두 값이 같을 수 있으며 `fingerprints`의 파일 종류별 이전/새 값과
`changes`를 함께 확인해야 한다.

변경이 발견되면 `SnapshotChangedWarning`과 함께 데이터셋별 변경 항목 및 이전/새
fingerprint를 비교 결과에 남긴다. 변경을 경고 없이 이전 값으로 교체하거나 미검증 상태를
`passed`로 올리지 않는다. 비교 결과는 원천 수정의 원인을 판단하는 입력이며 재수집
또는 데이터 사용 승인을 대신하지 않는다.

## 오프라인 예제

다음 4개 `CollectionResult`는 ETF·벤치마크·FX·Rf 역할을 보여주는 작은 합성 fixture다.
실제 API 응답·수집 실적·최종 유니버스·과거 공개시각 근거가 아니다. 실행하면 임시
디렉터리에만 기록하며 실제 15+4 데이터셋을 생성하지 않는다.

```python
from copy import deepcopy
from tempfile import TemporaryDirectory

import pandas as pd

from robo_advisor.data.collectors import CollectionResult
from robo_advisor.data.snapshot import compare_snapshots, read_snapshot, write_snapshot


def fixture(dataset_id, symbol, role, currency):
    days = pd.to_datetime(["2019-09-03", "2019-09-04"])
    if role == "risk_free":
        data = pd.DataFrame({"Value": [1.5, 1.6]}, index=days)
    else:
        data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [105.0, 106.0],
                "Low": [95.0, 96.0],
                "Close": [101.0, 102.0],
                "Volume": [1000, 1100],
            },
            index=days,
        )
    raw_data = data.copy(deep=True)
    raw_text = None
    if role == "risk_free":
        raw_data = pd.DataFrame(
            {"DATE": ["2019-09-03", "2019-09-04"], "DGS3MO": ["1.5", "1.6"]}
        )
        raw_text = "DATE,DGS3MO\n2019-09-03,1.5\n2019-09-04,1.6\n"
    issue = {
        "code": "SYNTHETIC_EXAMPLE",
        "severity": "unverified",
        "message": "Offline fixture; historical availability is absent",
    }
    metadata = {
        "dataset_id": dataset_id,
        "symbol": symbol,
        "source": "fred" if role == "risk_free" else "yahoo",
        "role": role,
        "market": "US_RATE" if role == "risk_free" else "FX" if role == "fx" else "US",
        "currency": currency,
        "timezone": "Asia/Seoul" if role == "fx" else "America/New_York",
        "unit": {
            "risk_free": "percent_per_annum", "fx": "KRW_per_USD",
            "benchmark": "index_points", "etf": "USD",
        }[role],
        "asset_class": "synthetic_fixture",
        "rationale": "Offline storage example; no team selection decision",
        "selection_status": "candidate" if role in {"etf", "risk_free"} else "confirmed",
        "requested_start": "2019-09-01",
        "requested_end_exclusive": "2019-09-05",
        "actual_start": "2019-09-03",
        "actual_end": "2019-09-04",
        "row_count": len(data),
        "collected_at": None,
        "price_basis": "Synthetic fixture; independent adjustment verification absent",
        "auto_adjust": None if role == "risk_free" else True,
        "status": "unverified",
        "issues": [issue],
        "availability_status": "unverified",
        "available_at": None,
    }
    validation = {
        "source": {"valid": False, "status": "unverified", "issues": [issue], "metrics": {}}
    }
    return CollectionResult(data, raw_data, metadata, validation, raw_text=raw_text)


results = {
    "SPY": fixture("SPY", "SPY", "etf", "USD"),
    "SP500": fixture("SP500", "^GSPC", "benchmark", "USD"),
    "USD_KRW": fixture("USD_KRW", "KRW=X", "fx", "KRW"),
    "DGS3MO": fixture("DGS3MO", "DGS3MO", "risk_free", "USD"),
}

with TemporaryDirectory(prefix="snapshot-demo-") as temporary_root:
    first = write_snapshot(results, root=temporary_root, snapshot_id="demo-a")
    restored = read_snapshot(first)
    assert set(restored.datasets) == set(results)
    assert restored.datasets["SPY"].status == "unverified"
    assert restored.datasets["USD_KRW"].metadata["available_at"] is None

    changed = deepcopy(results)
    changed["SPY"].data.iloc[-1, changed["SPY"].data.columns.get_loc("Close")] = 103.0
    second = write_snapshot(changed, root=temporary_root, snapshot_id="demo-b", previous=first)
    comparison = compare_snapshots(first, second)
    assert comparison["changed"][0]["dataset_id"] == "SPY"
    print(comparison)
```

실제 #9 결과를 사용할 때는 이미 확보한 `CollectionResult` mapping을 같은 API에
전달한다. 스냅샷 작성 과정에서 수집 함수를 다시 호출하지 않는다.

## 후행 #15 연결과 검증 범위

#15는 `read_snapshot(path).datasets`에서 복원한 자료를 받고 manifest의 파일 해시와
fingerprint를 lineage에 연결한다. 원천 snapshot ID, 가격 정의, 검증·가용성 상태도
함께 유지한다. `failed`/`unverified` 자료를 학습에 사용할지 결정하는 정책은 소비자
책임이며 저장 완료만으로 승인하지 않는다.

#12 정합·환산·미국 세션 래그·수익률 계산, 피처 버전·cutoff/warmup 기록과 최종
`features.parquet` 구성은 후행 계층이 담당한다. 실제 `available_at`이 없는 자료를
스냅샷의 생성시각으로 보완하지 않는다.

검증은 작은 mock `CollectionResult`를 사용해 저장/복원·원본 보존·결정적 fingerprint,
변경 분류·경고·손상 탐지·독점 작성과 기존 ID 보호를 확인한다. pyarrow 등 프로젝트
의존성 파일과 기존 `.venv`는 임의로 변경하지 않는다. 필요한 경우 승인된 별도 테스트
환경을 사용하며 실제 실행 환경과 결과는 검증 완료 후 기록한다.

## 최종 오프라인 검증 결과와 적용 상태 (2026-10-15)

프로젝트의 기존 Python 3.10.21 환경을 사용하고, 새 모듈은 허용된 Codex 작업 폴더에서
로드했다. Desktop 저장소의 새 파일 3개는 쓰기 권한 거부로 생성되지 않았다.
기존 소스·설정·테스트·문서·README·의존성 파일 57개의 SHA-256이 동일함을 확인했다.

| 검사 | 실제 결과 |
|---|---|
| Black | 기존 46개 + 신규 2개, 총 48개 파일 통과 |
| flake8 | 기존 src/tests/dashboard와 신규 2개 통과 |
| 신규 스냅샷 테스트 | 90 passed, 제외·실패 없음 |
| 기존 + 신규 전체 pytest | 638 passed, 기존 Gymnasium 경고 3건 |
| 누수 테스트 | 15 passed, 623 deselected, strict markers 사용 |
| 실제 Parquet 저장/읽기 | ETF·벤치마크·FX·금리 mock 왕복 통과 |
| 파일 SHA-256/내용 fingerprint | 손상 탐지, 동일 내용 비교, 값 변경 경고 통과 |
| 스냅샷 불변성 | ID 중복·기존 디렉터리 보호·부분 실패 정리·이전 파일 보존 통과 |

별도 경로에 사용자가 설치한 PyArrow 25.0.1을 테스트 runner의 모듈 검색 경로에만
추가했다. 프로젝트 소스에는 해당 임시 경로를 하드코딩하지 않았다. 실제 Parquet 엔진으로
원본·정규화 DataFrame, Yahoo MultiIndex 원본 열, CRLF 원문, 결측·역순·중복 관측,
빈 failed 자료와 실제 공개시각 Series를 왕복 검증했다. 지원하지 않는 원본 스키마는
명시적으로 거부하는 테스트도 통과했다. 공개시각 파일 없이 verified를 주장하는
manifest를 비교·읽기 양쪽에서 거부하는 누수 회귀 검사를 추가했다.

프로젝트 `.venv`와 requirements는 변경하지 않았다. pytest는 허용된 작업 폴더의
별도 `--basetemp`를 사용했다. 문서의 오프라인 예제도 실행했으며 기본 Windows Temp의
접근 오류에는 테스트 프로세스의 TEMP/TMP만 작업 폴더로 지정했다. Black은 동일 설정으로
각 파일을 순차 검사했다. Desktop 저장소 쓰기 제한은 남아 있으므로, 전달 파일을 실제
저장소에 적용한 뒤 같은 검사를 한 번 확인해야 한다.

작은 4역할 mock 스냅샷 2개를 `outputs/issue-11-mock-example/demo-a`, `demo-b`로
생성했다. 두 번째는 합성 SPY 종가만 바꾸었고 경고·이전/새 fingerprint 및 이전 스냅샷의
모든 파일 해시 불변을 확인했다. 이는 실제 시장 자료가 아니며 모든 상태는 unverified,
collected_at/available_at은 None이다. 실제 manifest는 각 디렉터리의 manifest.json,
비교 결과는 같은 예제 루트의 comparison.json을 참조한다.

실제 15개 투자자산 + 보조 데이터 4종의 스냅샷은 생성하지 않았다.
실제 공급자, 공식 거래소 캘린더, 독립 배당 조정 근거, 역사적 공개시각 검증은 남아 있다.
Git 명령·커밋·Push·PR·Merge, 실제 금융 데이터 다운로드와 robotics 원본 수정은
수행하지 않았다.
