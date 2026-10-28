# 금융 데이터 수집 계약 (#9)

> 이 문서의 후보 유니버스 설명은 `config/assets.yaml`에 해당한다. 확정 12자산의
> 별도 명세는 `config/assets_team12.yaml`이며, 최신 원천 검증 결과와 공개시각·가격
> 정의의 한계는 [원천 증거 검증](source-evidence.md)을 참조한다.

## 구현 범위와 현재 상태

`config/assets.yaml`과 `robo_advisor.data.collectors`는 데이터셋 명세를 읽고,
단일 Yahoo 티커의 일별 OHLCV 또는 FRED 금리 수준을 수집·검증·기록한다.
개인 `robotics/data/collect_missing_2019_2025.py`의 범용 Yahoo 호출,
단일 티커 열 정규화, FRED CSV helper에서 필요한 흐름을 재사용했다.
개인 파일을 import하거나 실행하지 않으며 원본을 변경하지 않는다.
기존 helper가 하던 잘못된 날짜/중복 삭제, 모델 기간 슬라이스, 임의 보간은 가져오지 않았다.

이 구현은 오프라인 mock으로 검증한다. 실제 데이터 재수집은 수행하지 않는다.
최종 12개 ETF와 Rf 정책이 미확정이므로 전체 유니버스의 확보 또는 #9 전체 완료를
선언하지 않는다. 설정에 있는 `confirmed`는 요구사항에 명시된 수집 대상이라는
뜻이며 과거 시점 가용성이나 원천 검증이 모두 통과했다는 뜻이 아니다.

## 설정과 미확정 팀 합의

설정 버전은 `schema_version: 1`이다. `load_asset_config(path)`가 YAML을 읽고
각 `datasets` 항목을 불변 `DatasetSpec`으로 변환한다. 데이터셋 필드는 아래와 같다.

| 필드 | 의미 |
|---|---|
| `dataset_id`, `symbol` | 팀 식별자와 공급자 티커/시리즈 ID |
| `source`, `role` | `yahoo`/`fred`, `etf`/`benchmark`/`fx`/`risk_free` |
| `market`, `currency`, `timezone` | 원천 시장, 가격 표시 통화, 일별 거래일 해석 시간대 |
| `asset_class`, `rationale` | 후보 자산군과 원천 존재/후보 근거 |
| `selection_status` | `candidate` 또는 `confirmed` |
| `unit` | 가격·지수·환율·금리의 원천 단위 |

ETF 후보는 미국 13개(SPY, IWM, EFA, EEM, IEF, TLT, TIP, LQD, GLD, VNQ,
QQQ, AGG, DBC)와 국내 3개(069500.KS, 278530.KS, 148070.KS), 총 16개다.
분석 보고서에서 확인한 후보 풀이며 투자 추천 또는 확정 12종 목록이 아니다.
`universe.target_etf_count`는 12, `status`는 `unconfirmed`, `selected_etfs`는 빈 목록이다.
069500.KS(KODEX 200)와 278530.KS(KODEX 200TR)는 별개 데이터셋이며 자동 대체하지 않는다.

`^KS11`, `^GSPC`, `KRW=X`는 Issue #9가 지정한 수집 대상으로 표시한다.
DGS3MO는 기존 원천에서 확인한 Rf 후보 하나다. 팀 Rf의 소스·시리즈·통화·만기,
연율에서 일별 수익률로의 환산식, 일수 기준은 모두 미확정이다.
`risk_free_policy`의 해당 값은 `null`이며 이 수집기는 정책을 결정하지 않는다.
공식 시장 캘린더 공급 경로, 과거 공개/이용 가능 시각 확보 방법,
ETF 배당 재투자 검증 근거와 가격지수 벤치마크의 배당 정의도 후행 합의가 필요하다.

## 공개 API와 기간

```python
load_asset_config(path) -> dict

collect_yahoo(
    spec,
    *,
    end,
    start="2019-09-01",
    download=None,
    trading_calendar=None,
    available_at=None,
    quote_metadata=None,
    allow_network=False,
) -> CollectionResult

collect_fred(
    spec,
    *,
    end,
    start="2019-09-01",
    http_get=None,
    trading_calendar=None,
    available_at=None,
    allow_network=False,
) -> CollectionResult
```

기본 시작일은 워밍업용 `2019-09-01`이다. 모든 어댑터는 `[start, end)`를 사용한다.
예를 들어 `end="2026-01-01"`은 2025-12-31까지의 관측을 요청한다.
시작일이 휴일이라는 이유로 평일을 만들어 넣거나 첫 관측까지 채우지 않는다.
FRED의 종료일 포함 조회 방식에는 `end - 1일`을 전달해 같은 배타적 계약을 맞춘다.
응답에 요청 범위 밖의 관측이 있으면 조용히 잘라내지 않고 실패 근거로 남긴다.
잘못된 `DatasetSpec` 또는 시작/종료일처럼 호출 전 설정 계약이 잘못된 경우에는
`ValueError`를 발생시킨다. 요청 이후 응답/API 오류는 결과의 `failed` 상태로 반환한다.

기본 `allow_network=False`이며 호출 가능한 mock을 주입하지 않은 호출은
`failed`와 `NETWORK_NOT_AUTHORIZED`를 반환한다. `download`와 `http_get`은
오프라인 테스트의 주입 지점이다. 임의의 실제 네트워크 함수를 주입해서 승인을
우회하지 않는다. 실제 API 사용은 별도 사용자 승인 후에만 `allow_network=True`로
실행하며, 프로젝트에 이미 지정된 yfinance/requests 의존성을 사용한다.

## Yahoo OHLCV와 가격 정의

국내 ETF는 `<코드>.KS`, 해외 ETF는 해당 Yahoo 티커를 사용한다. `^KS11`,
`^GSPC`, `KRW=X`도 같은 단일 티커 경로로 요청한다. Yahoo 호출은
`auto_adjust=True`와 일별 간격을 명시한다. 필수 가격 열은 Open/High/Low/Close이며
Volume도 함께 유지한다. 단일 티커 MultiIndex 응답은 요청한 티커와 price 열을
확인해 평탄화한다. 다중 티커나 모호한 열을 추측해 선택하지 않는다.

`auto_adjust=True`의 Close는 공급자의 배당·분할 조정 값이며, 별도 원종가나
독립적인 배당 재투자 지수가 아니다. 배당·분할 이벤트 열이 제공되면 함께 보존한다.
요청 옵션을 기록한 것만으로 배당 재투자 검증 통과를 선언하지 않는다.
현재 API는 상장일/최초거래일 및 독립 배당 재투자 비교 자료를 받지 않는다.
따라서 ETF의 상장일·배당 조정 검증은 항상 `unverified`이며, 이 최소 수집기만으로
ETF의 전체 결과를 `passed`로 승격할 수 없다.
`^GSPC`/`^KS11`의 지수 포인트와 ETF 조정가격은 서로 다른 가격 정의다.
지수 포인트를 USD/KRW 가격으로 가장하거나 ETF 가격을 지수로 바꾸지 않는다.

수집 결과의 일별 인덱스는 원천 시장의 현지 거래일 라벨로 해석한다.
UTC로 변환하면서 거래일 날짜를 바꾸지 않으며 원천 행 순서는 유지한다.
FX 시간대 `Asia/Seoul`은 이 설정의 일별 라벨 해석 기준이고 과거 환율의
공개 시각 또는 한국 종가 시각을 증명하지 않는다.

## 환율 방향과 금리 단위

`KRW=X`의 명세 단위는 **1 USD당 KRW**다. `unit="KRW_per_USD"`와
#12 연결용 `fx_direction="USD/KRW"`는 같은 의미를 다른 계약으로 표현한다.
`KRW/USD`라는 물리 단위 문자열을 그대로 #12 방향 인자에 전달하지 않는다.
[Yahoo 통화쌍 페이지](https://finance.yahoo.com/quote/KRW%3DX/)는 제공자 참조 링크다.
이번 작업에서 실제 Yahoo 통화쌍 응답을 수집해 방향을 인증한 것은 아니다.
가격이 1,000 이상이라는 값 크기 추정으로 방향을 검증하지 않는다.

방향 검증에는 호출자가 원천 근거를 확인한 뒤 정규화해서 제공한
`quote_metadata={"symbol": "KRW=X", "base_currency": "USD", "quote_currency": "KRW"}`를
사용한다. 메타데이터가 없으면 명세 방향은 기록하되 `fx_direction_status`를
`unverified`로 둔다. 원천 메타데이터가 반대 방향이거나 티커와 맞지 않으면 실패한다.
`fx_direction_status="verified"`는 호출자가 제공한 symbol/base/quote 값이
USD/KRW 스키마에 맞는다는 뜻이다. 수집기가 그 근거의 진위를 독립 인증한 상태가 아니다.
이번 mock 테스트의 인위적인 통화쌍 값은 실제 응답 검증 증거가 아니다.
기존 개인 DEXKOUS 파일을 KRW=X로 이름만 바꾸거나 동일 관측으로 취급하지 않는다.

FRED 어댑터는 [DGS3MO](https://fred.stlouisfed.org/series/DGS3MO)의
일별 관측을 `Value` 열, `percent_per_annum` 단위로 반환한다. 일별 관측 주기는
일별 수익률이라는 뜻이 아니다. 연율 수준을 100으로 나누거나 252/365로 환산하지
않으며 이미 환산된 `rf`라고 표시하지 않는다. 유한한 0 또는 음수 금리는 허용한다.
FRED의 `.` 결측 표현은 의미상 결측으로 변환하지만 관측 행은 보존하고 검증에 남긴다.
해석 불가능한 값이나 Inf를 결측으로 조용히 바꿔 버리지 않는다.
수집기 API는 합의된 Rf 정책을 입력받지 않으므로 Rf 정책 검증은 항상
`unverified`다. 이후 팀 설정에 정책이 명시돼도 이 어댑터가 일별 환산을 수행하지는 않는다.

## 결과·원본 보존·검증

`CollectionResult`는 아래 값을 반환하며 저장하지 않는다.

| 값 | 계약 |
|---|---|
| `data` | 검증 대상 정규화 DataFrame; 현지 일별 라벨과 행 순서 보존 |
| `raw_data` | 공급자 DataFrame의 깊은 사본; 날짜/열/행 순서/이상 값 보존. Frame 확보 전 실패 시 `None` |
| `raw_text` | FRED 원문 CSV 문자열. 파싱 실패 시에도 확보한 원문 보존; Yahoo 결과에서는 `None` |
| `metadata` | 명세, 요청·실제 기간, 행 수, 수집시각, 가격 정의와 상태 |
| `validation` | 기존 #10 검증 보고서 및 수집 어댑터 검증 보고서 |
| `status` | `metadata["status"]`의 `passed`/`failed`/`unverified` |

메타데이터에는 출처와 출처 URL, 티커/시리즈, 역할·시장·통화·시간대·단위,
선정 상태, 요청 시작/종료일과 종료일 배타성, 실제 최소/최대 관측일,
단일 `row_count`, 현재 UTC `collected_at`, `auto_adjust` 옵션 및 `price_basis`를
기록한다. 행 삭제/추가를 하지 않으므로 확보한 원천 Frame의 행 수를 `row_count`로
기록한다. Yahoo `source_url`은 티커 quote 페이지이며 FRED는 graph CSV URL이다.
`requested_end_exclusive` 키가 종료일 계약을 표현한다. `collected_at`은 응답을
확보한 이번 호출의 수집시각이며 역사적 공개시각이 아니다. 응답을 확보하지 못한
경우에는 `None`이다. `actual_start`/`actual_end`는 날짜가 모두 해석 가능할 때만
최소/최대 관측일을 기록하며 잘못된 날짜가 하나라도 있으면 둘 다 `None`이다.
이 기간은 휴일/누락 세션 완전성 인증이 아니다.

#10 원천 검증기의 가격/날짜, 거래 세션, 조정가격 관련 검증을 연결한다.
금리에는 양수 가격 검사를 적용하지 않고 별도 숫자·결측 검증을 사용한다.
어댑터는 필수 열, OHLC 관계와 Volume, 요청 기간, 공급자 스키마, 환율 방향,
실제 `available_at` 입력 등의 수집 계약도 확인한다. 검증 결과를 평균 내거나
기존 개인 프로젝트의 PASS 값으로 대체하지 않는다.

| 상태 | 해석과 소비자 처리 |
|---|---|
| `failed` | 빈 응답, API 예외, 날짜/열/가격 결함 등 오류가 존재. 오류 보고서 확인 후 소비 정책 결정 |
| `unverified` | 검사한 오류는 없지만 후보 선정, 공식 달력, 과거 가용성, 배당 근거 등 확인되지 않은 항목 존재 |
| `passed` | 수행한 모든 필수 검사와 제공된 검증 근거가 통과한 경우에만 사용 |

오류가 하나라도 있으면 전체 상태는 `failed`다. 오류가 없더라도 미검증 사유가
있으면 `unverified`다. 중복일 삭제, 역순 정렬, 결측/비양수 행 삭제, Forward Fill,
백필, 원천 이상의 자동 보정은 수행하지 않는다. 공급자가 DataFrame을 반환했다면
실패하더라도 원천 사본을 남긴다. HTTP 실패처럼 아직 DataFrame이 없는 경우에는
빈 결과와 실패 보고서를 반환하며 가짜 관측을 만들지 않는다.

`trading_calendar`는 호출자가 별도로 확보한 시장 캘린더다. 관측 날짜 또는 단순
월~금 목록을 실제 거래소 캘린더로 추정하지 않는다. 달력이 없으면 세션 완전성을
`unverified`로 남긴다. Yahoo의 응답 행이 적다는 사실을 조용히 정상으로 처리하지 않는다.

`available_at`은 원천 관측일에 대응하는 실제 시간대 포함 Series를 받는다.
날짜 라벨이 원천의 모든 행과 같은 순서로 대응하는지, 값이 시간대 포함 시각인지,
공개시각이 비감소하는지, 해당 관측일의 현지 자정 이후인지 검증한다.
통과한 값은 UTC로 변환해서 `metadata["available_at"]`에 보존하고
`availability_status="verified"`로 기록한다. 이 상태는 입력 계약 검증 결과다.
실제 공개시각의 근거는 호출자 책임이며 진위·빈티지·관측시점별 이용 가능 여부의
독립 인증은 후행 작업이다. 누락 시 미검증으로 기록하고, 잘못된 입력도 근거 보존을
위해 사본을 남긴다.
현지 종가 시각, 다음 날 자정, 조회 시각으로 과거 공개시각을 생성하지 않는다.
최신 FRED 조회본의 관측 날짜는 역사적 공개일 또는 당시 빈티지를 뜻하지 않는다.

## 오프라인 실행 예

아래 응답은 테스트용 인위적인 값이다. 다운로드나 파일 저장을 수행하지 않는다.

```python
from pathlib import Path

import pandas as pd

from robo_advisor.data.collectors import collect_yahoo, load_asset_config

config = load_asset_config(Path("config/assets.yaml"))
spec = next(item for item in config["datasets"] if item.dataset_id == "SP500")

mock_frame = pd.DataFrame(
    {
        "Open": [100.0, 101.0],
        "High": [102.0, 103.0],
        "Low": [99.0, 100.0],
        "Close": [101.0, 102.0],
        "Volume": [1000, 1100],
    },
    index=pd.to_datetime(["2019-09-03", "2019-09-04"]),
)


def mock_download(tickers, **kwargs):
    assert tickers == "^GSPC"
    assert kwargs["auto_adjust"] is True
    return mock_frame.copy(deep=True)


result = collect_yahoo(spec, end="2019-09-05", download=mock_download)
assert result.status == "unverified"  # 과거 가용성·시장 캘린더 근거 미제공
assert len(result.raw_data) == 2
```

컬렉터 단위 테스트는 `download`/`http_get` mock과 네트워크 금지 경로를 검증한다.
기존 팀 코드와 잠금 의존성 파일을 수정하지 않는다. 프로젝트의 Python 3.10 환경에서
아래 검증을 실행한다. pytest 임시폴더 권한 오류에는 사용자 Temp 아래의 별도
`--basetemp` 경로를 사용한다.

```powershell
.\.venv\Scripts\python.exe -m black --check src tests dashboard
.\.venv\Scripts\python.exe -m flake8 src tests dashboard
.\.venv\Scripts\python.exe -m pytest tests/test_collectors.py -v
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m pytest -m leakage -v --strict-markers
```

## 실제 오프라인 검증 결과 (2026-10-13)

프로젝트 `.venv`의 Python 3.10.21에서 다음 검증을 완료했다.

| 검증 | 결과 |
|---|---|
| Black `--check src tests dashboard` | 통과, 46개 파일 변경 불필요 |
| flake8 `src tests dashboard` | 통과, 종료 코드 0 |
| `tests/test_collectors.py -v` | 86 passed |
| 전체 pytest | 548 passed, 3 warnings |
| `-m leakage -v --strict-markers` | 12 passed, 536 deselected |

전체 테스트의 경고 3건은 기존 Gymnasium 환경의 Box 공간 권고다.
수집기 테스트는 모든 소켓 연결을 차단하고 Yahoo/FRED 응답을 mock으로 주입했다.
실제 금융 데이터 재수집은 수행하지 않았다. YAML을 실제로 로딩·검증하기 위해
사용자 승인 후 잠금 버전 PyYAML 6.0.3만 `--no-deps`로 설치했다. 다른 의존성 버전과
requirements 파일은 변경하지 않았다.

최초 전체 실행은 기존 설정 테스트의 `monkeypatch.chdir` 복원 시 Codex 실행 환경의
저장소 폴더 접근 권한 때문에 종료했다. 해당 기존 파일을 수정하지 않고, 권한이 있는
Codex 작업 폴더에서 같은 프로젝트 `.venv`와 절대 테스트 경로를 사용해 다시 실행했다.
설정 테스트 36개와 전체 548개가 통과했다. 이 실행에는 작업 폴더 아래의 별도
`--basetemp` 및 `-p no:cacheprovider`를 사용했다. 재현한 전체 검증 명령은 다음과 같다.

```powershell
# 권한이 있는 Codex 작업 폴더에서 실행
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:/Users/FORYOUCOM/Desktop/NitroRogueClone/Robo-Advisor/.venv/Scripts/python.exe' `
    -m pytest 'C:/Users/FORYOUCOM/Desktop/NitroRogueClone/Robo-Advisor/tests' `
    -q -p no:cacheprovider `
    --basetemp='C:/Users/FORYOUCOM/Documents/Codex/2026-10-12/files-pasted-by-the-user-robo/work/pytest-issue9-full-2'
```

기존 소스·설정·테스트·문서·잠금 의존성 파일 51개의 작업 전후 SHA-256을 비교했고
변경이 없었다. 작성한 저장소 파일은 요청된 4개뿐이다. `.git/HEAD` 직접 읽기로
`feat/9-market-data-collectors`를 확인했으며 Git 명령, 커밋/Push/PR 조작은 하지 않았다.

## #11·#12·#15 연결

- **#11 스냅샷:** `raw_data`, `data`, `metadata`, `validation`을 인계한다. 저장 계층에서
  failed/unverified를 명시적으로 처리하고 원천·검증 계보를 함께 저장한다. Parquet 저장,
  manifest 및 파일 해시는 #11이 담당하며 수집기는 이 기능을 구현하지 않는다.
- **#12 정합·수익률:** 별도 구현이 병합된 이후 현지 거래일과 조정가격, 실제
  `available_at`, 공식 시장 캘린더, 환율 `fx_direction="USD/KRW"`를 전달한다.
  미국 종가 래그, KRX/미국 교집합, KRW 환산과 로그수익률 계산은 #12에 맡긴다.
  수집시각을 역사적 가용성 시각으로 대체하거나 #9에서 미리 래그/수익률을 계산하지 않는다.
- **#15 입력 데이터셋:** #11 스냅샷의 명세·검증과 #12 이후의 정합·가용성 계보를
  유지해서 `features.parquet` 및 lineage에 원천 해시, 피처 버전, cutoff/warmup을
  기록한다. 피처 생성과 저장은 #15에서 구현한다. 후보 전체를 확정 action asset으로
  선택하거나 연율 금리를 일별 Rf로 자동 연결하지 않는다. 학습/검증 분할과 정규화는
  후행 계층에서 처리한다.

[PR #82](https://github.com/team-broke/Robo-Advisor/pull/82)의 최신 요구사항 추적표는
연계 작업과 산출물의 진행 상태를 판단하는 참조다. 2026-10-13 확인한 head
`738f28d098ed2672fb8734131d94a033dd54988e`는 develop의 #10 원천 검증기와
미구현 #9/#11/#15, 진행 중 #12를 구분한다. #82 자체도 open/unmerged다.
[PR #81](https://github.com/team-broke/Robo-Advisor/pull/81)의 #12 구현은 열린 PR의
계약 참조이며 확인한 head `c03df105f6c0b28475a03ea6097240f783a9dba8`도
open/unmerged다. develop에 포함됐다고 가정하지 않는다. 이 #9 모듈은 미병합 #12 코드를
import하지 않는다. 실제 데이터 재수집, Git 커밋/Push, PR 생성/Merge는 이번 작업에 포함하지 않는다.
