# Issue #12: 거래일·로그수익률 전처리 계약

`robo_advisor.data.alignment`은 수집기 없이 pandas 입력만으로 동작한다.
한국 자산의 입력 단위는 KRW, 미국 자산/벤치마크의 입력 단위는 USD다.
수익률은 **현재 공통 평가일과 직전 공통 평가일 사이의 후행 로그수익률**이다.
다음 날 reward, 합성 벤치마크의 가중 합성, 지표, 정규화, 스냅샷/데이터셋 생성은
이 모듈의 범위에 포함되지 않는다.

## 입력과 캘린더

- `domestic_prices_krw`, `overseas_prices_usd`: 자산을 열로 가진 비어 있지 않은
  DataFrame. 두 시장의 자산 이름은 겹치지 않아야 한다.
- `kr_calendar`, `us_calendar`: **실제 KRX/미국 거래 세션**의 DatetimeIndex.
  호출자가 거래소/공급자의 캘린더 출처, 범위, 완전성을 보장한다. 이 모듈은
  평일 달력이나 가격 관측 행을 거래일 달력으로 대체하지 않는다.
- 모든 가격/캘린더 날짜는 #10의 현지 거래일 계약으로 정규화한다. 한국은
  `Asia/Seoul`, 미국은 `America/New_York`를 사용하며 출력 라벨은 naive 날짜다.
  중복, 역순, NaT, naive/aware 혼합, 중복 열을 거부한다. 오류를 숨기기 위해
  정렬하거나 중복을 삭제하지 않는다. 가격이 캘린더 밖에 있으면 검증 실패다.
- 캘린더에는 처리할 원천 기간의 **모든 세션**이 필요하다. 평가 시작 이전의
  미국 세션을 포함하면 래그 시드로 사용할 수 있다. 범위 밖 시드는 달력 경계
  제외 사유로 구분하고 휴장으로 잘못 기록하지 않는다.
- `fx_rates`: 환율 Series. 관측 날짜 라벨은 서울 현지 날짜 계약이다. 날짜의
  의미가 다른 공급자 자료는 호출자가 이 계약에 맞게 명시적으로 변환해야 한다.
- `fx_direction`: 필수. `USD/KRW`는 1 USD당 KRW, `KRW/USD`는 1 KRW당 USD다.
  역방향은 명시적으로 역수 변환한다. 다른 표기는 거부한다. 수치 크기만으로
  잘못 표시된 실제 단위를 검출할 수는 없으므로 공급자의 단위 확인은 호출자 책임이다.
- `fx_available_at`: 필수 Series. 환율과 동일한 관측 날짜 index 및 순서, NaT 없는
  timezone-aware **실제 공개/이용 가능 시각**을 요구한다. 공개 시각은 비감소
  순서이며 관측일의 서울 00:00보다 앞설 수 없다. 관측일만으로 당일 사용을
  추정하지 않는다. FRED의 날짜만 있고 공개 시각이 불확실하면 호출 전에 근거
  있는 보수적인 이용 가능 시각을 정해야 하며, 없으면 이 API는 실행을 거부한다.
- `domestic_available_at`, `overseas_available_at`: 선택 Series. 시장별 입력 가격의
  관측 날짜에 대응하는 실제 공개 시각을 지정할 수 있다. 기본값은 한국 서울
  15:30, 미국 뉴욕 16:00이다. 미국은 DST를 반영하며 조기 폐장일에도 16:00을
  사용하므로 보수적이다. 공급자 지연이나 다른 종가 정의가 있으면 명시적 시각을
  전달해야 한다. 전달된 시각의 진위나 공급자/공식 달력의 완전성은 자동 인증하지 않는다.

## 처리 순서와 결측 정책

1. #10 `validate_price_data`로 원천 날짜/가격을 검증한다. 양수인 실수만 허용한다.
   0, 음수, Inf, 문자열 숫자, bool 등은 실패하며 `AlignmentError.reports`에 원천
   보고서를 보존한다. NaN만 아래의 명시적인 결측 정책으로 처리한다. 원천 보고서의
   `failed`를 `passed`로 바꾸지 않고 결측 처리 전후 보고서를 별도로 기록한다.
   WTI의 음수 현물 가격 등은 이 가격 로그수익률 경로에 넣지 않아야 한다.
2. 각 시장의 **전체 원천 세션 축**에 reindex한다. 누락 행과 셀 NaN 모두 #10
   `validate_missing_intervals`로 검사한다. `max_forward_fill=5`가 기본값이며,
   그보다 긴 연속 결측은 전체 실행을 실패시킨다. 비율은 통계로 기록한다.
   앞 구간을 포함해 한도를 초과하면 실패한다. 0은 결측 없는 입력에만 허용한다.
3. 시장별 가격을 최대 한도까지 forward fill하며 원천 날짜, 실제 이용 가능 시각,
   채움 플래그도 함께 이동한다. 선행 NaN은 유지한다. backward fill은 없다.
   교집합 이전에 처리하므로 평가하지 않는 원천 거래일도 유효한 과거 정보로
   사용할 수 있다. 시점 검사는 아래 단계에서 원천의 실제 공개 시각으로 수행한다.
4. 미국 전체 거래 세션에서 가격/계보를 `shift(1)`한다. 달력 날짜 하루를 빼거나
   공통 평가일 목록에서 shift하지 않는다. 미국 월요일은 직전 미국 금요일 종가를
   쓰며 미국 휴장일은 세션으로 세지 않는다. 최초 세션에는 유효한 과거값이 없다.
5. 공통 평가일은 두 시장 달력의 교집합이고 평가 시각은 서울 **15:30**이다.
   국내/래그된 해외 가격 중 실제 공개 시각이 평가 시각보다 늦으면 해당 셀을
   결측으로 둔다. 이후 가격을 다시 fill하지 않는다.
6. 환율은 평가 시각까지 공개된, 결측이 아닌 마지막 관측을 선택한다. 공개 시각이
   평가 시각과 같으면 사용한다. 환율 NaN 구간은 #10으로 관측 행 단위의 연속
   결측 한도를 검사한다. 누락 행/비정기 공개에 무제한으로 과거 환율을 쓰지 않도록
   `max_fx_age=pd.Timedelta(days=7)`로 공개 시각과 관측일 시작으로부터의 나이를
   **둘 다** 제한한다. 한도 초과 시 그 평가일 환율은 결측이고 `FX_STALE`로 기록한다.
   이전 공개값을 사용하는 경우 관측일/공개 시각과 채움 여부를 보존한다.
7. 해외 USD 가격에 선택된 USD/KRW를 곱한다. 국내 가격에는 환율을 곱하지 않는다.
   가격 forward fill을 **환산 전에** 하는 이유는 해외 USD 가격을 유지하더라도
   공개된 환율 변동을 현재 원화 가격에 반영하기 위해서다. 환산 overflow/underflow는
   검증 실패다. USD 벤치마크 열에도 동일한 래그/환율 정책을 적용할 수 있다.
8. 잔여 결측을 가진 공통 날짜 축에서 `log(price).diff()`를 계산한다. 수학적으로
   `log(price_t / price_previous)`와 같고 극단적인 가격 비율 overflow를 피한다.
   첫 행/미보정 결측 행/그 직후 행은 수익률이 없으며 0으로 바꾸지 않는다.
   결측 행을 먼저 drop하면 여러 평가일에 걸친 수익률이 생기므로 계산 후 제외한다.
9. 가격과 모든 자산 수익률이 완전한 행만 함께 반환하고 NaN/Inf가 없는지 확인한다.
   정보가 부족하면 빈 결과와 진단을 반환한다. 가격 carry로 실제 같은 가격이 된
   경우의 0 수익률은 `forward_filled`로 구분한다. 수익률 자체는 fill하지 않는다.

입력 DataFrame/Series/캘린더는 변경하지 않는다. 최종 행 제외가 앞선 수익률 계산을
바꾸지 않으므로 반환 수익률의 기준 날짜는 `return_start_dates`로 확인해야 한다.
첫 반환 행의 기준 가격은 워밍업 때문에 반환 가격 표에 없을 수 있다.

## 함수와 반환 결과

| 함수 | 입력 | 반환 |
| --- | --- | --- |
| `align_trading_calendars` | 두 시장 캘린더 | 공통 거래일 DatetimeIndex |
| `lag_overseas_prices` | USD DataFrame, 전체 US 캘린더, 선택 공개 시각/채움 한도 | `LaggedPrices`: 가격, 원천 날짜, 공개 시각, 채움 플래그, 원천/결측 진단 |
| `convert_prices_to_krw` | 같은 평가 날짜로 시점 정합된 USD 가격과 환율, 필수 방향 | KRW DataFrame; 날짜가 다르면 실패, 시점 선택은 파이프라인 책임 |
| `calculate_log_returns` | 증가하는 날짜의 양수 가격 DataFrame | 동일 축의 수익률 DataFrame; 첫 행과 잔여 결측 전후 NaN 유지 |
| `preprocess_market_data` | 두 시장 가격, 환율, 캘린더, 환율 공개 시각/방향 | `AlignmentResult` |

`AlignmentResult`의 표/Series는 모두 최종 공통 index에 대응한다.

- `prices_krw`, `log_returns`: 국내 열 다음 해외 열 순서. NaN/Inf 없음.
- `source_dates`, `available_at`, `forward_filled`: 자산별 실제 원천 날짜,
  UTC aware 이용 가능 시각, 가격 채움 여부.
- `fx`: `rate`(항상 USD/KRW), `observation_date`, UTC aware `available_at`,
  `forward_filled`(평가 날짜보다 과거의 환율 관측을 사용했는지).
- `evaluation_at`: UTC aware 한국 평가 시각.
- `return_start_dates`: 수익률의 직전 공통 평가일.
- `diagnostics`: 정책/가정, #10 원천 보고서, 시장별 원천 채움 전후 결측 통계,
  원천/평가 축 가격 채움 횟수, 환율 과거 관측 사용 횟수, 제외 날짜와 사유,
  공통/제외/최종 행 수, 정합 가격 결측 셀 수, 최종 NaN/Inf 검사 결과.

대표 제외 사유는 `KRX_CLOSED`, `US_CLOSED`, `OUTSIDE_*_CALENDAR_RANGE`,
`PRICE_NOT_AVAILABLE`, `MISSING_ALIGNED_PRICE`, `FX_NOT_AVAILABLE`, `FX_STALE`,
`RETURN_WARMUP`, `MISSING_PREVIOUS_PRICE`다. 한 날짜에 복수 사유가 있을 수 있다.
원천 전체 세션의 채움 횟수와 공통 평가일에서 사용한 채움 횟수는 축이 다르다.

```python
import pandas as pd
from robo_advisor.data.alignment import preprocess_market_data

sessions = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
result = preprocess_market_data(
    pd.DataFrame({"KR_ETF": [100.0, 110.0, 121.0]}, index=sessions),
    pd.DataFrame({"US_ETF": [10.0, 20.0, 40.0]}, index=sessions),
    pd.Series([1000.0, 1100.0, 1200.0], index=sessions),
    kr_calendar=sessions,  # 실제 공급자 캘린더의 이 기간 세션
    us_calendar=sessions,
    fx_available_at=pd.Series(
        (sessions + pd.Timedelta(hours=10)).tz_localize("Asia/Seoul"),
        index=sessions,
    ),  # 예시에서 명시한 실제 공개 시각
    fx_direction="USD/KRW",
)
# 1/4 가격: KR_ETF=121, US_ETF=20*1200=24000.
# US 원천 날짜=1/3, 수익률 기준일=1/3.
```

#13은 `prices_krw`의 자산 Series로 지표를 계산할 수 있다. #14는 해당
Walk-Forward Train 범위에서만 fit하며, #15는 반환 날짜/계보/진단을 명시적으로
이어받아야 한다. 기존 지표/정규화기/분할기 구현은 변경하지 않는다.

## v5와의 차이 및 검증 범위

읽기 전용으로 확인한 v5 메타데이터는 2020-01-02~2025-12-30, 1,507행,
128피처, 15 Action Assets를 기록한다. 미국 Master Calendar에서 한국 자산을
as-of 정렬하고 한국 휴장 가격을 유지하는 정책이다. 이번 출력은 KRX/US 교집합,
전체 미국 세션 래그, 공개 시각이 확인된 FX를 사용하므로 v5와 행 수/가격/수익률의
동일성을 목표로 하지 않는다. v5의 next-return reward도 이번 후행 수익률과 다르다.
원본 ZIP/CSV는 수정하거나 저장소에 추가하지 않았다.

단위/누수 테스트는 작은 pandas 자료와 명시적 캘린더로 외부 API 없이 실행한다.
실제 공급자의 휴장 캘린더 정확성, FRED 공개 이력, v5 대용량 가격과의 수치 회귀는
검증하지 않았다. 수집기 #9의 단위/공개 시각/캘린더 계약 확정 후 연결 검증이 필요하다.
