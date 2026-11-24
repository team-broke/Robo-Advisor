# Issue #27 벤치마크 데이터 검증

감사기는 KOSPI와 S&P 500의 원천 종가, 참조 거래 세션, 평가시각, FX 관측을
대조한다. 두 지수의 수익률과 날짜별 사유를 반환한다. 합성 비중·재조정과
백테스트는 #1에서 팀 합의를 거쳐 적용하는 후속 단계다.

## 보존 자료와 확인 범위

| 입력 | 확인 | 미확인 |
| --- | --- | --- |
| KOSPI | Yahoo Finance 레거시 CSV의 티커는 ^KS11, 단위는 지수 포인트, 기준통화는 KRW. 원문 1,717행(2019-01-02~2025-12-30), 모델 기간 KRX 참조 1,473세션 중 1,471행. | 두 결측일 공식 종가, 전체 기간의 KRX 종가 대조, 최초 요청·원응답 및 당시 공개시각. |
| S&P 500 | 별도 국면 연구 CSV 1,684행(2020-01-02~2026-09-15), 모델 기간 US 참조 1,508세션에 모두 관측. 처음 네 종가를 FRED SP500과 소수 둘째 자리까지 대조. | CSV의 실제 제공자·티커·요청·원응답·공개시각. 저장소의 ^GSPC 수집 명세만으로 연구 CSV 출처를 인정할 수 없다. |
| SMBS MAR 및 ALFRED DEXKOUS | 두 시나리오는 각각 KRW per USD. 보존 입력 해시와 #106의 FX as-of 재현 로직을 사용한다. | 실제 날짜별 공개시각. SMBS는 다음 날 10:00 서울, ALFRED는 선언한 빈티지 이후 사용 시각이라는 기존 연구 가정이 남아 있다. |
| Yahoo KRW=X | 부모 원천은 OHLC 모순 156일, 모든 OHLCV 값 결측 4일로 failed. | 공급자 내부 오류 원인. 이 원천은 이 감사의 FX 시나리오에 사용하지 않는다. |
| 시장 일정 | XKRX/US 참조 세션과 평가시각을 별도 파일로 사용한다. | 전 기간 공식 휴장·조기 폐장·특별 개폐장과 공급자별 종가 가용시각의 독립 인증. |

연구 CSV 두 개의 Close와 Adj Close는 일치한다. 배당 재투자 여부의 증빙으로는
충분하지 않다. 현재 보존 후보는 PR 가정으로 검사한다. 명세가 섞인 PR/TR
조합은 실패하며, TR 입력에는 해당 지수의 명시적인 공식 계열 식별자가 필요하다.
KOSPI 200 TR, KODEX 200TR, SPY는 두 원지수와 서로 다른 계열이다.
두 지수는 기존 부모·선택 스냅샷의 구성원이 아니다.

공식 비교 경로는 [KRX 지수 시세](https://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd),
[FRED SP500 종가](https://fred.stlouisfed.org/series/SP500),
[S&P DJI 지수 산식](https://www.spglobal.com/spdji/en/documents/methodologies/methodology-index-math.pdf)이다.
FRED SP500은 S&P DJI가 제공하는 배당 미포함 가격지수다. KOSPI TR의 기간별
원자료와 S&P 500 TR의 공식 과거 레벨은 별도로 확보해야 한다.

FX 정의와 공개 이력은 [SMBS](https://www.smbs.biz/ExRate/StdExRate.jsp),
[DEXKOUS](https://fred.stlouisfed.org/series/DEXKOUS),
[H.10 발표 안내](https://www.federalreserve.gov/releases/h10/about.htm),
[ALFRED 빈티지](https://alfred.stlouisfed.org/help)에서 확인한다. 시장 일정은
[KRX 휴장일](https://open.krx.co.kr/contents/MKD/01/0110/01100305/MKD01100305.jsp)과
[NYSE 일정](https://www.nyse.com/trade/hours-calendars)의 과거 공지를 대조한다.

## 날짜별 결측과 산식

| 대상 | 가격 또는 FX 이용 불가 평가일 | 인접 수익률에 영향 |
| --- | --- | --- |
| S&P 500 초기 미국 종가 | 2020-01-02에 필요한 2019-12-31 관측이 연구 CSV에 없음 | 2020-01-02, 2020-01-03 |
| KOSPI | 2022-01-03, 2022-05-09 | 2022-01-03, 2022-01-04, 2022-05-09, 2022-05-10 |
| SMBS FX | 2023-10-04(184시간), 2025-01-31(184시간), 2025-10-10(208시간). 한도 168시간 | 2023-10-04, 2023-10-05, 2025-01-31, 2025-02-03, 2025-10-10, 2025-10-13 |
| ALFRED FX | 모델 기간 1,424 공통 평가일 중 연구 가정하에 사용 불가 0일 | 이 기간 FX로 인한 추가 부재 없음 |

보존 자료를 이 감사기로 재실행한 결과, 공통 참조 평가일 1,424일 중
인접 수익률 계산이 가능한 날짜는 SMBS 1,412일과 ALFRED 1,418일이다.
두 시나리오의 원천 감사 결과는 모두 unverified이며 training_approved는 false다.

미국 종가는 전체 미국 일정에서 공통 평가일의 직전 세션을 찾는다. 종가
이용 가능 시각을 입력에서 제공하면 그 시각을 쓴다. 없으면 참조 세션 종료
시각을 가정하고, 결과 행의 timing_reported는 false로 기록한다. 달력 CSV는
미국 휴장·조기 폐장과 DST를 반영한 시각을 제공해야 한다.

FX는 관측 기준시각 및 이용 가능 시각이 평가 컷오프 이하인 원천 관측을 선택한다.
실제 공개시각이 제공되면 연구 가정보다 우선한다. S&P 500 원화 환산 레벨은
직전 미국 종가 × 해당 평가일의 유효 FX다. 두 지수 모두 직전 공통 평가점
대비 현재 레벨 비율에서 1을 빼 단순수익률을 계산한다. 두 끝점에 필요한
모든 값이 있어야 두 수익률을 반환한다. 관측 부재 또는 시각 초과는
exclusion_reasons와 affected_return_dates에 남으며 가격 보간은 적용하지 않는다.
FX 경과시간 초과는 FX_STALE과 fx_observation_age_hours로 확인한다.

## 로컬 재현

원천 파일을 공통 증빙 루트 아래에 두고, 별도의 JSON 설정에서 각 파일의 기존
SHA-256과 상대 경로를 지정한다. 아래 파일명은 예시다. 보존 파일을 증빙
루트에 읽기 전용으로 매핑하거나 별도 작업 폴더에 복사해도 된다.

~~~json
{
  "schema_version": 1,
  "start": "2020-01-01",
  "end_exclusive": "2026-01-01",
  "benchmarks": {
    "KOSPI": {
      "path": "KOSPI_2019_2025.csv",
      "sha256": "7046e78e3cb3ec308e7790f7a7d9d801a5e41dc25e8be8fb122680d2d6d73d1e",
      "index_id": "KOSPI", "source": "yahoo", "symbol": "^KS11", "currency": "KRW",
      "return_type": "PR"
    },
    "SP500": {
      "path": "SP500.csv",
      "sha256": "d49607aed66d66904e3c098a1c44a5579ad78f4f795e0912326629da491ebc8f",
      "index_id": "SP500", "source": "unknown", "currency": "USD", "return_type": "PR"
    }
  },
  "calendars": {
    "KRX": {
      "path": "krx-reference-sessions-20190901-20251231.csv",
      "sha256": "a39cd8fb075a73db300dd0b25c247c9be2fb6c8f333ada66450b296560bf673a"
    },
    "US": {
      "path": "us-sessions-20190901-20251231.csv",
      "sha256": "88f6fb8d828bce278ab0649f5312758d4ee6bf8baa284bc64d2ef74dd21dd314"
    }
  },
  "evaluation": {
    "path": "date-eligibility-all.csv",
    "sha256": "183acf66633dc13cfe90b546d604a715e4e0c6c3bb56972f96611169adf468c2"
  },
  "fx": {
    "path": "smbs-usdkrw-mar-20190901-20251231.csv",
    "sha256": "126b120c6a566114cbeb31f224e8e7a268a02b69d879d3eead563d9370a8295d",
    "source": "smbs", "max_age_days": 7
  }
}
~~~

- 지수 CSV는 Date와 Close가 필요하다. Yahoo 레거시의 Ticker와 Source를
  제공하면 선언값과 대조한다. available_at은 타임존을 포함한 선택 열이다.
- 캘린더 CSV는 session_date와 타임존 포함 close 또는 scheduled_close_utc, 평가 CSV는 session_date와
  evaluation_at을 제공한다. 평가 범위 내 모든 공통 세션이 순서대로 있어야 한다.
- ALFRED 재현 시 FX·평가 파일을 함께 바꾼다. 두 시나리오의 파일 해시와
  경로는 기존 team12_audit.json을 참조한다. 보존 ALFRED 연구 한도는 14일이다.
- 선택 사항 claims에 별도 해시 고정 CSV를 제공하면 전체 평가일의
  session_date, return_start_date, sp500_source_date, fx_observation_date,
  fx_krw_per_usd, sp500_krw, kospi_simple_return, sp500_krw_simple_return을
  재계산과 대조한다. claims_check의 passed는 해당 열의 일치를 뜻한다.
- 선택 사항 comparison은 전략의 유효 수익률 구간 CSV다. session_date와
  return_start_date가 감사 범위의 유효 벤치마크 구간과 정확히 같아야 한다.
  개별 윈도우에 대해서는 validate_comparison_periods에 해당 구간의 두 시작일
  Series를 전달할 수 있다. Series 인덱스가 수익률 종료일이다.

~~~text
python -m robo_advisor.data.benchmark_audit --root "<evidence-root>" --config "<audit-config>.json"
python -m pytest -q tests/test_benchmark_audit.py
python -m pytest -q -m leakage --strict-markers
python -m pytest -q
python -m black --check src tests dashboard
python -m flake8 src tests dashboard
~~~

CLI는 프로젝트 패키지를 설치한 환경에서 실행한다. 설치 없이 실행하는
PowerShell 환경에서는 현재 세션의 PYTHONPATH에 저장소 src를 지정한다.
CLI는 JSON을 표준 출력으로 반환한다. 종료 코드 1은 입력·대조 실패, 2는
미확인 원천이 있는 감사다. 해시 일치와 계산 재현은 원천의 신원·공개시각
독립 인증을 대신하지 않는다. 이 감사에서는 training_approved를 항상 false로
기록한다. 단위 테스트의 작은 시계열은 합성 입력이다.

## Issue #1 팀 합의

- 두 지수의 PR/TR 및 적용할 배당 재투자·세금 기준을 전략 성과의 정의와 맞춘다.
- KOSPI/S&P 500(KRW)의 50:50 비중에 적용할 재조정 주기·시점과 공통 비용 규칙을 확정한다.
- 평가 컷오프, FX 원천·경과시간 한도·헤지 여부, 실제 공개시각의 증빙 기준을 확정한다.
- 평가 날짜와 수익률 양 끝점, 초기 기준점, 결측일·다음 수익률 처리 기준을 윈도우별로 고정한다.
