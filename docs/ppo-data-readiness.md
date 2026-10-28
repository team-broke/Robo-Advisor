# Issue #94 연동 준비: 피처 산출물과 PPO 입력 계약

이 문서는 최신 `develop`의 #15 피처 빌더와 #91 PPO 실행기를 대조한 결과다.
#94 변환기나 PPO 환경은 구현·수정하지 않는다. 형식 대응이 가능하다는 사실은
실제 원천 검증 완료 또는 학습 승인을 의미하지 않는다.

## 값과 파일의 대응

| #15 산출물 | 의미 | PPO 입력 계약과 필요한 대응 |
|---|---|---|
| 루트 `features.parquet`: `ASSET__log_return` | 시점 t에서 끝나는 후행 로그수익률, 워밍업은 NaN | `returns.parquet`: 같은 구간의 단순수익률 `expm1(log_return)` |
| 루트 `features.parquet`: `ASSET__rsi14` | 원척도 RSI, 0~100 | `rsi.parquet`: 같은 값. 환경이 관측 구성 시 100으로 나눔 |
| 루트 `features.parquet`: `ASSET__macd` | `(EMA12 - EMA26) / 현재 가격` | `macd.parquet`: 가격 정규화 MACD 선. signal·histogram·z-score로 대체하지 않음 |
| `ASSET__macd_signal`, `ASSET__macd_histogram`, `ASSET__volatility20` | 72열 계약에 포함되는 추가 지표 | 현재 PPO 환경 입력 세 파일에는 직접 대응 없음 |
| 루트 `rewards.parquet` | t행에 기록된 다음 평가 구간 t→t+1 로그수익률 | 후행 `returns[t]`와 시간축이 다름. 파일 이름을 바꾸거나 관측으로 사용하지 않음 |
| `windows/<id>/features.parquet` | 해당 Window Train에만 fit한 72열 z-score | 위 세 PPO 파일의 원척도 입력으로 그대로 사용하지 않음 |
| `windows/<id>/rewards.parquet` | 분할·워밍업·불연속 경계 마스크가 적용된 미래 target | 변환 검증용 별도 target 근거. 환경의 후행 returns와 혼합하지 않음 |

MACD의 가격 정규화와 Train scaler의 z-score 정규화는 서로 다른 연산이다.
로그수익률에 `expm1`을 적용할 때도 z-score 전의 루트 값을 사용해야 한다.
예를 들어 로그수익률 `log(1.1)`은 단순수익률 `0.1`에 대응한다.
PortfolioEnv는 시점 t에 리밸런싱한 뒤 `returns[t+1]`을 적용한다.
이미 t+1을 가리키는 #15 target을 returns[t]로 넣으면 미래 정보가 관측에 섞인다.

## 고정 자산·피처 순서와 시각

고정 자산 순서는 `SPY, IWM, EFA, EEM, IEF, TIP, LQD, GLD, DBC, VNQ,
KODEX_200TR, KTB10Y`다. #15 루트 파일은 자산 우선으로 각 자산의
`log_return, rsi14, macd, macd_signal, macd_histogram, volatility20` 순서를
유지한다. PPO 세 파일은 각각 이 자산 순서의 12열이어야 하며 인덱스도 완전히
같아야 한다. 환경 행동은 12자산 다음 현금의 13개 가중치다.

#15의 인덱스는 시간대 없는 일자 라벨이다. PPO는 중복·NaT 없이 오름차순인
시간대 포함 DatetimeIndex를 요구한다. 변환 담당자는 `rows.parquet`의 검증된
`evaluation_at`을 일자와 일대일로 대응시켜 세 프레임에 동일하게 적용해야 한다.
자정에 UTC를 붙이거나 모든 일자에 현재 거래시간을 생성하는 방식으로 역사적
평가시각을 대신하지 않는다. 실제 공개시각과 연구용 공개 가정도 별도로 보존한다.

#15의 strict 계약은 현재 서울 15:30 평가를 검사한다. 별도 연구 브랜치의
동적 폐장시각 정책을 현재 develop에 병합된 기능으로 간주하지 않는다.
실제 일정의 특수 폐장일과 연구용 가정이 이 계약과 다르면 명시적으로 기록하고
연동 전에 정책 담당자와 합의해야 한다.

## 워밍업·불연속·Walk-Forward 경계

- #15는 각 연속 구간의 첫 33행을 워밍업으로 마스킹한다. `feature_ready`,
  `segment_id`, `return_interval_contiguous`, `next_date`와 Window의
  `eligible_observation`, `reward_eligible`을 보존한다.
- NaN 행을 없앤 뒤 서로 다른 연속 구간을 단일 PPO 에피소드로 붙이면 단절을
  가로지르는 수익률·lookback이 생길 수 있다. 구간별 충분한 과거 문맥과 실제
  수익률 구간을 확인하고 부족한 구간을 제외한 근거를 기록해야 한다.
- PortfolioEnv의 모든 returns는 유한해야 한다. RSI·MACD는 첫 관측 이전
  `lookback-1`행의 워밍업만 허용한다. 첫 관측 이후 NaN을 0이나 미래값으로
  채우지 않는다. 최소 행 수와 에피소드 길이 조건도 별도로 확인해야 한다.
- #15 경계는 `[start, end)`이고 Train/Validation/Test 모두 유효한 관측이 필요하다.
  PPO `data.train_end`는 포함 경계다. 일자 경계를 실제 evaluation_at으로
  대응시키고 마지막 포함 관측을 명시해야 한다. 날짜 문자열을 그대로 복사하면
  경계 행이 바뀔 수 있다.
- PPO의 `validation.start`는 학습 컷오프 안에서 학습 환경과 검증 환경을 다시
  나누며 검증용 관측 문맥 `lookback-1`행만 빌린다. #15 Window 분할을 자동으로
  읽는 기능은 없다. #94와 #25에서 Window ID·각 경계·문맥 정책을 연결해야 한다.
- Window별 Train 평균·scale과 ddof=0, fit 행 수·기간을 계보에 유지한다.
  Validation/Test로 refit하지 않는다. 현재 PPO 계약은 z-score된 RSI나
  z-score된 수익률을 요구하지 않는다.

현재 PPO는 72열 전체를 그대로 관측하는 구조가 아니다. 자산 수 A와 lookback N에
대해 관측은 수익률 이력, 보유비중, RSI/100, MACD, risk_score로
`A*(N+4)+1` 차원이다. 예를 들어 A=12, N=20이면 289차원이다.
추가 지표를 사용하려면 별도 환경·모델 계약 변경이 필요하다.

## 계보와 학습 승인

변환 담당자는 원천 snapshot manifest SHA-256, 원래 lineage SHA-256과 파일 해시,
asset/feature 순서, 원척도와 변환식, 날짜↔평가시각 매핑, Window ID·경계,
워밍업·제외 행·제외 사유, 변환 버전과 새 세 파일의 SHA-256을 연결해야 한다.
기존 `read_snapshot`과 `read_feature_dataset`의 해시 검사를 먼저 재사용하고
원본·기존 출력 폴더를 덮어쓰지 않는다.

현재 PPO 로더는 입력 파일의 바이트 해시를 실행 manifest에 남기지만 #15 lineage,
source validation, `training_approved` 또는 분할 마스크를 자동으로 검사하지 않는다.
따라서 형식이 맞고 로더가 성공했다는 이유로 학습 승인 여부를 확정할 수 없다.
외부 원천·공개시각·배당 조정·체결 정책의 미확인 항목은 계속 차단 사유다.
이번 후속 검증은 원척도·시점·해시·누수 상태를 보고하며 `training_approved=false`를
유지한다. 실제 학습 또는 변환기 완성을 주장하지 않는다.

## 담당자에게 넘길 최소 검증 항목

1. `expm1` 값 대응과 12자산 열 순서, RSI 범위, MACD 선의 가격 정규화 정의.
2. 루트 미래 target의 관측 혼입 금지, t→t+1 보상 대응과 마지막 미확정 행.
3. 일자와 실제 evaluation_at 일대일 대응 및 알려진 미래 공개값 차단.
4. Window별 경계·문맥·불연속 처리, Validation/Test가 Train fit에 영향 없는지 확인.
5. 원천·lineage·새 PPO 파일 해시 연결과 failed/unverified 자료의 승인 우회 차단.
6. 소규모 합성 변환 회귀와 실제 자료 검증을 구분한 뒤 별도 합의로 학습 실행.

## 코드 근거

- `src/robo_advisor/features/dataset.py`: `_FEATURES`, `_validated_alignment`,
  `_window_features`, `build_feature_dataset`.
- `src/robo_advisor/features/dataset_io.py`: `read_feature_dataset` 및 원척도·해시 계약.
- `src/robo_advisor/features/indicators.py`: `calculate_rsi`, `calculate_macd`.
- `src/robo_advisor/rl/training.py`: `DataConfig`, `load_training_dataset`, `run_training`.
- `src/robo_advisor/rl/environment.py`: `PortfolioEnv` 입력 검증·관측·step 계약.
- `src/robo_advisor/rl/validation.py`: `split_validation`.
- `tests/test_training.py`: 입력 형식 오류·컷오프 뒤 변경에 대한 누수 회귀.

관련 작업: [Issue #94](https://github.com/team-broke/Robo-Advisor/issues/94).
이 문서는 담당자의 변환기를 대체하지 않는다.
