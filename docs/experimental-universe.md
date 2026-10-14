# v5 15자산 실험 유니버스

## 목적과 범위

이 설정은 기존 개인 v5의 15개 action asset을 같은 순서로 참조하는
`v5_15` 실험 프로필이다. 팀의 정식 **12개 ETF 목표**나 최종 선정 합의를 대체하지 않는다.
`config/assets.yaml`의 16개 ETF 후보 풀과 선정 상태를 유지한 채 별도 설정에서
실험에 사용할 ID를 선택한다. 기존 README, 수집기, 환경, 의존성 파일을 변경하지 않는다.

추가 파일은 아래 4개다.

- `config/experiment-v5-15.yaml`: 별도 실험 프로필 및 후행 정합 계약
- `src/robo_advisor/experiments/universe.py`: 명세 로딩·mock 수집·감사·환경 입력 연결
- `tests/test_experimental_universe.py`: 외부 API 없이 계약과 동작 검증
- `docs/experimental-universe.md`: 본 문서

실데이터 수집, 스냅샷 저장, 가격 정합·환산, 수익률·피처 생성, PPO 모델 생성·학습은
구현하지 않는다. 이번 단계는 소스 명세와 후행 입력의 연결을 검증하는 단계다.

## 실험 목록과 역할

투자자산의 열 순서는 아래와 같이 고정된 설정 순서다. 수집 결과의 dict 순서,
가격·RSI·MACD 입력 열, 환경의 자산 및 행동 순서에서도 같은 순서를 사용한다.

| 순서 | dataset_id | 원천 티커 | 시장·통화 |
|---:|---|---|---|
| 1 | SPY | SPY | US · USD |
| 2 | IWM | IWM | US · USD |
| 3 | EFA | EFA | US · USD |
| 4 | EEM | EEM | US · USD |
| 5 | IEF | IEF | US · USD |
| 6 | TLT | TLT | US · USD |
| 7 | TIP | TIP | US · USD |
| 8 | LQD | LQD | US · USD |
| 9 | GLD | GLD | US · USD |
| 10 | VNQ | VNQ | US · USD |
| 11 | QQQ | QQQ | US · USD |
| 12 | AGG | AGG | US · USD |
| 13 | DBC | DBC | US · USD |
| 14 | KODEX_200TR | 278530.KS | KRX · KRW |
| 15 | KTB10Y | 148070.KS | KRX · KRW |

15개 모두 등록 풀의 `selection_status="candidate"`를 그대로 유지한다.
`KODEX_200`/069500.KS는 `excluded_dataset_ids`로 표시한다. 278530.KS의 대체
티커로 바꾸지 않으며 기존 등록 풀에서 삭제하지 않는다.

보조 데이터는 KOSPI/^KS11과 SP500/^GSPC 벤치마크 2개, USD_KRW/KRW=X 환율 1개,
DGS3MO Rf 후보 1개다. 투자자산 15개와 보조 데이터 4개를 합쳐 수집 계획은 **19개**다.
보조 자료는 투자자산 열과 action asset에 포함하지 않으며 환경의 자산별 수익률·RSI·MACD
입력에서도 제외한다. FX는 해외 가격의 후행 KRW 환산, 벤치마크는 비교 성과,
Rf 후보는 팀 정책 검토에 사용한다. DGS3MO를 최종 Rf로 확정하거나 연율 값을
일별 수익률로 연결하지 않는다.

## 설정 로딩과 수집 API

```text
load_experimental_universe(path) -> ExperimentalUniverse

collect_universe(
    universe,
    *,
    end,
    yahoo_download=None,
    fred_get=None,
    trading_calendars=None,
    available_at=None,
    fx_quote_metadata=None,
) -> dict[str, CollectionResult]

audit_universe(
    universe,
    results,
    *,
    end,
    trading_calendars=None,
    listing_dates=None,
    dividend_evidence=None,
) -> dict

make_portfolio_env(
    universe,
    simple_returns,
    rsi,
    macd,
    *,
    lookback,
    **env_options,
)
```

`load_experimental_universe("config/experiment-v5-15.yaml")`는 설정 파일 기준의
상대 경로 `asset_config: assets.yaml`을 해석하고 기존 #9 로더의 `DatasetSpec`을
참조한다. ID 존재 여부·중복·역할, 제외 대상과 투자 대상의 충돌, 설정 개수,
정합 정책과 워밍업 시작일을 검증한다. 임의의 티커 목록을 별도로 하드코딩하거나
참조한 후보를 `confirmed`로 승격하지 않는다.

로더와 환경 연결 함수는 설정으로부터 자산 수를 계산한다. 제공 프로필의
`expected_asset_count: 15`와 실제 목록 길이가 일치하는지 확인하며, 범용 로더 코드가
모든 실험을 무조건 15개로 강제하는 것은 아니다. 정식 등록 풀의 12개 목표 규칙은 유지한다.

`ExperimentalUniverse.asset_ids`는 순서 있는 tuple이며 `collection_datasets`는
투자자산·벤치마크·FX·Rf 후보를 순서대로 제공한다. `dimensions(lookback)`으로
환경 차원을 확인한다. `lookback`은 양의 정수여야 한다.

`collect_universe`는 dataset_id별 `CollectionResult`를 반환한다. 기본 실제
Yahoo/FRED 클라이언트는 호출하지 않으며 #9 수집기에 `allow_network=False`를 전달한다.
mock을 제공하지 않으면 각 수집기는 `NETWORK_NOT_AUTHORIZED` 실패 결과를 반환한다.
주입한 함수가 오프라인 mock이라는 보장은 호출자의 책임이다. 실제 클라이언트를
주입해 별도 수집 승인을 우회하는 용도로 사용하지 않는다.

기간은 #9 계약에 따라 **`[2019-09-01, end)`**다. `trading_calendars`와
`available_at`은 각각 dataset_id를 키로 하는 mapping이며 실제 근거를 전달한다.
관측 날짜나 단순 평일 목록으로 공식 시장 캘린더를 만들어 넣지 않는다.
`fx_quote_metadata`는 #9에서 정의한 KRW=X의 symbol/base/quote 근거다.
메타데이터 일치 검증과 원천 근거의 독립 인증은 구분한다.

## 워밍업·원천 감사

`audit_universe`는 요청 전체 `[2019-09-01, end)`의 실제 기대 세션을 담은 외부
`trading_calendars`, 실제 ETF `listing_dates`, 독립 `dividend_evidence`를 받아
#10 검증기를 추가 호출한다. 입력 mapping에 실험 외 ID가 있으면 거부한다.

감사 결과는 `investment_assets`와 `supporting_datasets`의 벤치마크·FX·Rf 후보를
구분한다. 데이터셋별 원천 상태, 요청·실제 기간과 행 수, 가격 정의·가용성 상태,
세션 누락/결측 검사, ETF 상장일 및 조정가격 검사, issue 목록을 기록한다.
수집 결과의 데이터와 원본, 메타데이터, 기존 검증 상태를 수정하지 않는다.

| 상황 | 감사 해석 |
|---|---|
| 결과가 없음 | `NO_COLLECTION_RESULT`, `unverified`; 실제 수집 가능성 또는 확보를 선언하지 않음 |
| 원천 오류가 있음 | 원래 오류를 유지하여 `failed` |
| 요청·데이터셋 메타데이터가 계획과 다름 | 계약 불일치 `failed` |
| 달력·상장일·독립 배당 근거가 없음 | 해당 검증 `unverified` |
| 근거를 추가했으나 원천에 미검증 사유가 남음 | 원천 issue를 유지; 전체 `passed`로 자동 승격하지 않음 |

외부 달력으로 누락 세션·결측 비율·연속 결측 구간을 점검한다. 원천에 존재하는
날짜만 검사해서 워밍업 완전성을 선언하지 않는다. 상장일 검사는
`require_full_period=True`로 전체 요청 기간의 확보 가능성을 확인한다.
`dividend_evidence[dataset_id]`에는 #10 계약의 `raw_close`, `reinvested_close`,
`dividends`, `trading_calendar` 같은 실제 비교 근거를 제공한다. 감사 함수가
공급자 가격이나 배당 이벤트로 독립 비교 자료를 합성하지 않는다.

Yahoo 요청의 `auto_adjust=True`는 공급자 수정주가 사용 옵션이다. ETF의
독립적인 배당 재투자 검증 통과와는 별개다. #9 수집기는 상장일·배당 비교 근거를
입력받지 않아 그 검증을 `unverified`로 반환한다. 실험 감사가 추가 검증에 성공하더라도
그 원천 보고서를 지우지 않는다. Rf 정책 및 후보 선정의 미확정 상태도 유지한다.

원천 결측·중복일·역순·비정상 값은 삭제하거나 채우지 않는다. `CollectionResult.raw_data`와
FRED의 `raw_text` 보존 계약은 #9와 같다. 실제 결과가 없는 상황에서는 기존 개인
파일의 존재나 과거 PASS 문자열만으로 현재 19개 명세의 수집/검증 완료를 선언하지 않는다.

## #12 정합 계약과 환경 입력

실험 설정의 `alignment_policy`는 아래 4개 필드를 그대로 유지해야 한다.

```yaml
base_currency: KRW
fx_direction: USD/KRW
overseas_close_lag_sessions: 1
require_available_at: true
```

KRW=X의 단위는 **1 USD당 KRW**이며 #12 방향 표기는 `USD/KRW`다.
해외 종가는 먼저 **전체 미국 거래 세션에서 1세션 래그를 적용한 뒤** KRX/미국
거래일 교집합에 연결한다. 교집합을 만든 후 단순히 한 행을 이동하면 같은 정책이 아니다.
실제 시간대 포함 `available_at` 근거가 필요하며 역사적 시각을 현재 수집시각,
관측일 자정 또는 임의의 종가 시각으로 만들어 넣지 않는다.

이 모듈은 설정을 검사할 뿐, 교집합·래그·FX as-of 환산·수익률 계산을 수행하지 않는다.
미병합 #12 코드를 import하지 않으며 실제 후행 정합은 #12가 통합된 이후 호출자가
담당한다. #12의 현지 거래일 라벨과 최종 환경 입력의 UTC 시각 계약도 구분한다.

`make_portfolio_env`는 각 입력 DataFrame의 열이 `universe.asset_ids`와 **순서까지
정확히 일치**하는지 확인한다. 누락·추가·재정렬 또는 벤치마크/환율/Rf 열 혼입을
자동 보정하지 않는다. `n_assets`는 투자 목록 길이에서 도출하며 호출자가
`env_options`로 덮어쓰는 것을 거부한다. 인덱스 동일성, 시간대 포함·증가·고유 시각,
유한한 값과 RSI 범위 등은 기존 `PortfolioEnv`가 검증한다.
이 함수는 수집/감사 보고서를 입력받지 않는다. 실제 실험에 미검증 원천을 사용할지
판단하고 후행 정합 계보를 확인하는 책임은 호출자에게 있다.

환경은 시점 t에 끝나는 **simple return**을 받는다. #12에서 반환한 로그수익률은
정합과 KRW 환산이 끝난 후 호출자가 `np.expm1(log_returns)`로 변환해야 한다.
이 연결 함수는 가격이나 로그수익률을 simple return으로 변환하지 않는다.
naive 일별 라벨에 UTC 시간대를 붙이는 것만으로 실제 평가 시각을 만들지 않는다.
호출자는 원천 가용성과 정합 계보에 근거한 실제 `evaluation_at`을 확보하여 세 입력을
동일한 시간대 포함 평가 인덱스로 연결해야 한다. 최종 환경은 이 인덱스를 UTC로 변환한다.
`evaluation_at` 확보와 인덱스 변환도 이 모듈의 구현 범위 밖이다.

15자산 프로필의 행동 차원은 **16**이다. 설정 순서의 자산 15개 뒤에 현금이
마지막 원소로 붙는다. 행동은 기존 환경의 softmax 대상 logits다.
관측 차원은 `A × (N + 4) + 1`이며 A=15일 때 아래와 같다.

| lookback N | n_assets | action_dim | observation_dim |
|---:|---:|---:|---:|
| 2 | 15 | 16 | 91 |
| 20 | 15 | 16 | 361 |

기존 환경의 관측은 N개 수익률, 자산/현금 비중, RSI, MACD, 자산별 risk score를
포함한다. raw 수집 데이터나 보조 자료 4개를 이 열 구조에 직접 넣지 않는다.

## 오프라인 예제

다음 값과 날짜는 **테스트용 합성 자료**다. 실제 수집 가능성, 워밍업 완전성,
가격·공개시각 인증 또는 v5 데이터 재현을 의미하지 않는다.

```python
import numpy as np
import pandas as pd

from robo_advisor.experiments.universe import (
    audit_universe,
    collect_universe,
    load_experimental_universe,
    make_portfolio_env,
)

universe = load_experimental_universe("config/experiment-v5-15.yaml")
days = pd.to_datetime(["2019-09-03", "2019-09-04", "2019-09-05"])
mock_prices = pd.DataFrame(
    {
        "Open": [100.0, 101.0, 102.0],
        "High": [102.0, 103.0, 104.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [101.0, 102.0, 103.0],
        "Volume": [1000, 1100, 1200],
    },
    index=days,
)


def mock_yahoo(**kwargs):
    assert kwargs["auto_adjust"] is True
    return mock_prices.copy(deep=True)


class MockFredResponse:
    text = "DATE,DGS3MO\n2019-09-03,1.5\n2019-09-04,1.6\n2019-09-05,1.7\n"

    def raise_for_status(self):
        pass


def mock_fred(url, **kwargs):
    return MockFredResponse()


results = collect_universe(
    universe, end="2019-09-06", yahoo_download=mock_yahoo, fred_get=mock_fred
)
assert len(results) == 19
report = audit_universe(universe, results, end="2019-09-06")
assert report["status"] == "unverified"  # 실제 캘린더·공개/상장/배당 근거 없음

# 환경 연결만 검사한다. 위 가격에서 수익률을 생성하는 예제가 아니다.
mock_evaluation_at = pd.date_range("2025-01-01 06:30", periods=3, freq="D", tz="UTC")
columns = universe.asset_ids
simple_returns = pd.DataFrame(0.001, index=mock_evaluation_at, columns=columns)
rsi = pd.DataFrame(50.0, index=mock_evaluation_at, columns=columns)
macd = pd.DataFrame(0.0, index=mock_evaluation_at, columns=columns)
env = make_portfolio_env(universe, simple_returns, rsi, macd, lookback=2)
observation, info = env.reset()
assert observation.shape == (91,)
assert env.action_space.shape == (16,)
env.step(np.zeros(16))
```

예제의 `mock_evaluation_at`은 합성 fixture다. 실제 데이터에는 진짜 평가 시각과
후행 정합 결과가 필요하며 이 인덱스를 복사해 공개시각 근거로 사용하지 않는다.

## 기존 v5 자료의 근거와 차이

원본은 `C:/Users/FORYOUCOM/Desktop/robotics/data/roboadvisor_v5_rl_dataset.zip`과
`roboadvisor_features_v5_128.zip`이다. 2026-10-15에 ZIP 중앙 목록과 작은 JSON/Markdown,
CSV 헤더 및 가격 CSV 첫 관측 한 행만 읽었다. CSV 전량 재검증과 joblib
역직렬화는 수행하지 않았다. 관련 기존 분석은 `issue-9-reuse-analysis.md`와
`issue-9-data-inventory.md`에 기록되어 있다.

15개 순서는 RL ZIP 내부 `action_assets_v5.json`, `metadata_v5_rl.json.action_assets`,
가격·next return·PPO의 열 순서, WF1/WF2 metadata의 assets에서 모두 일치한다.
`asset_prices_v5.csv`의 2020-01-02 첫 관측은 15개 모두 100이다.

| 기존 v5 기록 | 이번 실험과의 관계 |
|---|---|
| 각 자산 2020-01-02=100 가격지수 | 자산 순서·회귀 비교 근거; 실제 KRW 원천가격으로 복사하지 않음 |
| 미국 master date에 한국 과거 as-of 가격 연결, 휴장일 수익률 0 | 팀 #12의 교집합·미국 종가 래그 계약과 다르므로 그대로 적용하지 않음 |
| State 2020-01-02~2025-12-30, 마지막 reward 2025-12-31 | 기존 메타데이터의 기록; 이번 mock 기간의 수집 실적이 아님 |
| 128피처 및 원천 PASS 보고서 | 당시 자료의 주장; 현재 #10 통과 및 #15 최종 피처 증거가 아님 |
| 2019 원천 워밍업과 과거 공개값 사용 정책 설명 | 실제 전체 캘린더·역사적 시각을 인증한 자료와 구분 |

기존 ZIP 루트 CSV 헤더에는 `available_at`/공개시각/수집시각/cutoff/source hash 열이
없다. `FEATURE_DICTIONARY_v5_128.md:308`의 공개된 과거 값 사용 설명과
`metadata_v5_128.json`의 backward/as-of 정책 문구는 역사적 시간대 포함 시각 증거가 아니다.
`v5_alignment_report.csv`에는 국내 ETF 각 107개 as-of 채움과 최대 stale 7일을 기록한다.
`v5_validation_report.csv`는 환율 결측 78개·WTI 결측 76개에도 PASS를 기록한다.
이를 현재 검증에서 조용히 승인하거나 결측을 자동 삭제하는 근거로 사용하지 않는다.

기존 metadata의 시장 국면은 `provisional_until_team_protocol_lock`다.
KOSPI/S&P500(KRW) 50:50, 60 master dates와 ±5% 기준도 팀 최종 정책으로 확정하지 않는다.
Rf 소스·통화·만기·일별 환산/일수 정책 및 최종 팀 ETF 12종은 여전히 합의가 필요하다.

## 통합 상태와 검증 범위

2026-10-15에 확인한 [PR #83](https://github.com/team-broke/Robo-Advisor/pull/83)은
open/unmerged이며 head는 `843cd6672dd2d006a50b3f5ea07bf5982aa97330`이다.
이를 develop에 병합된 코드로 가정하지 않는다.
[PR #81](https://github.com/team-broke/Robo-Advisor/pull/81)의 head
`c03df105f6c0b28475a03ea6097240f783a9dba8`도 미병합 #12 참조다.
본 실험 모듈은 미병합 #12 코드를 import하지 않는다.

#11의 Parquet·manifest/hash 저장, #12의 정합·환산·수익률, #15의
`features.parquet` 및 source hash/피처 버전/cutoff/warmup 계보는 후행 작업이다.
실험 명세·수집 결과·감사 보고서와 순서 있는 자산 ID를 그 계층에 인계한다.

검증은 외부 API 없는 mock 수집, 원천 감사, 설정 로딩, 환경 reset/step 및 차원 계약을
대상으로 한다. 현재 Python 3.10 `.venv`에는 stable-baselines3/torch가 없으므로
실제 PPO forward는 미검증이다. 사용자 승인에 따라 의존성은 그대로 유지한다.
Gym 환경의 행동·관측 차원 검증을 PPO 모델 실행 통과로 표현하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m black --check src tests dashboard
.\.venv\Scripts\python.exe -m flake8 src tests dashboard
.\.venv\Scripts\python.exe -m pytest tests/test_experimental_universe.py -v
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m pytest -m leakage -v --strict-markers
```

## 실제 검증 결과 (2026-10-15)

지정 저장소 `.venv`의 Python 3.10.21에서 검증했다.

| 검사 | 결과 |
|---|---|
| Black 전체 `src/tests/dashboard` | 48개 파일 통과 |
| flake8 전체 `src/tests/dashboard` | 종료 코드 0 |
| `test_experimental_universe.py` | 52 passed |
| 전체 pytest | 600 passed, 3 warnings |
| 누수 `-m leakage -v --strict-markers` | 13 passed, 587 deselected |

경고 3건은 기존 Gymnasium 환경의 Box 공간 권고다. 실험 테스트는 소켓 연결을
차단하고 18개 Yahoo 요청과 1개 FRED 요청을 mock으로 검증했다. 정책·계약 불일치,
원천 결함 보존, 독립 조정가격 근거 전달, 15열의 순서·개수 및 환경 reset/step을 확인했다.

Black 다중 파일 실행은 Windows 제한 환경의 내부 `socketpair`에서 대기했다.
프로젝트 Black 설정과 `--check`를 그대로 사용하면서 파일 하나씩 순차 실행해
전체 48개를 검사했다. Black·네트워크·의존성 설정 파일은 변경하지 않았다.
pytest는 Codex 작업 폴더에서 프로젝트 Python과 절대 테스트 경로로 실행하고,
작업 폴더 아래의 실행별 `--basetemp` 및 `-p no:cacheprovider`를 사용했다.

실제 금융 데이터 다운로드나 실제 공급자 연결 성공을 검증한 것은 아니다.
15개 ETF의 현재 전체 가격 가용성, 공식 캘린더 완전성, 배당 조정·공개시각 근거는
실제 원천을 승인받아 확보한 이후 별도로 확인해야 한다. `audit_universe`는 그 검사를
수행할 입력 경로를 제공하며, 이번 mock 통과를 실제 원천 `passed`로 표시하지 않는다.
PPO forward는 사용자 승인에 따라 미검증으로 유지했고 의존성을 설치하지 않았다.

기존 소스·설정·테스트·문서·README·의존성 파일 57개의 작업 전후 SHA-256은 동일하다.
기존 `config/assets.yaml`과 README의 12개 목표를 보존했으며 요청된 새 파일 4개만 추가했다.
Git 상태는 작업 시작 시 `feat/9-market-data-collectors`에서 clean임을 확인했다.
커밋·Push·PR 생성·Merge 및 robotics 원본 수정은 수행하지 않았다.
