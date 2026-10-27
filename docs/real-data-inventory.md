# 실제 12자산 연구 데이터 실측 인벤토리

2026-10-27 KST 로컬 파일을 읽기 전용으로 확인했다. 외부 금융 API나 공식 사이트를 새로 조회하지 않았고, 원천·스냅샷·기존 산출물을 수정하지 않았다. 파일 보존·수치 재계산을 확인한 결과와 과거 보고서의 증빙 주장을 구분한다. 모든 자료의 `training_approved=false`를 유지한다.

## 위치와 재현 기준

기존 연구 산출물은 저장소 밖의 `team12-experimental-<delivery-id>` 패키지에서 발견했다. 아래 `<evidence-root>`는 사용자가 보존한 그 패키지의 루트다. 개인 PC 절대 경로를 코드나 설정에 넣지 않는다. 저장소의 `config/team12_audit.json`은 이 루트에 대한 상대 경로와 기존에 기록한 SHA-256을 사용한다.

- 부모 원천: `inputs/parent-snapshot/20261016T121929355317Z_cec9d76f691f41f9b8bda1772a056091/`
- 연구 선택 원천: `results/actual-research/source-snapshots/20261017T032943682306Z_b55bdb6e38c24f3cbfa638f702cc08b0/`
- 실제 연구 산출물: `results/actual-research/{smbs,alfred}/datasets/actual-observed-experiment/`
- 원천 FX: `evidence/fx-korea/smbs-usdkrw-mar-20190901-20251231.csv`, `evidence/fx-fred/ALFRED-weekly-vintage-observations.csv`
- 연구 일정: `inputs/calendars/krx-reference-sessions-20190901-20251231.csv`, `inputs/calendars/us-sessions-20190901-20251231.csv`
- 날짜별 선택·제외: `results/actual-research/{smbs,alfred}/selected-observed-FX.csv`, `date-eligibility-all.csv`
- 원천 채택 이력: `results/actual-research/source-adoption-decisions.csv`

원천의 대용량 Parquet, HTTP 원문, Excel 및 개인 경로는 Git에 추가하지 않는다. 파일이 없는 환경에서는 자료 없음으로 보고하고 합성 자료로 대체하지 않는다. 합성 fixture는 API 검증에만 사용한다.

## 이번에 직접 확인한 해시

| 자료 | SHA-256 |
|---|---|
| 부모 manifest | `4b0de83602496a6cc976c6b9d057b4dd6823ffd76fb6c71b5f1c733e4bb793f0` |
| 연구 선택 manifest | `52c79c8acf010df607c824bede4448436bcafdb46066aecbcb7777aef087382e` |
| SMBS CSV | `126b120c6a566114cbeb31f224e8e7a268a02b69d879d3eead563d9370a8295d` |
| ALFRED CSV | `eba081d2d4720b0651f57542bad26a70c4327c013ab037a047fb62808d3ac44c` |
| SMBS lineage | `7a07c022284779252852000af61ffb998542274ed9dd0949ff8a7044e3211dfa` |
| ALFRED lineage | `9cdf2fdd1c0174b8e9d4fe25db49a4363c94c1fa35ec0804a5a127e0a7bd3e0b` |

`read_snapshot(..., expected_manifest_sha256=...)`로 부모 13원천 및 선택 14원천의 무결성 검사를 통과했다. 두 연구 데이터셋 각 9개, 총 18개 Parquet의 바이트 SHA-256을 `lineage.files`와 직접 비교해 전부 일치했다. 이 해시들은 보존된 감사 기록과의 동일성을 뜻한다. 공급자의 진실성, 과거 시점 가격 빈티지 또는 공식 전송 인증까지 증명하지 않는다.

## 12자산 원천과 실제 결함

| 자산 | 티커 | 통화 | 실제 보유 기간 | 행 수 |
|---|---|---|---|---:|
| SPY | SPY | USD | 2019-09-03~2025-12-31 | 1,592 |
| IWM | IWM | USD | 동일 | 1,592 |
| EFA | EFA | USD | 동일 | 1,592 |
| EEM | EEM | USD | 동일 | 1,592 |
| IEF | IEF | USD | 동일 | 1,592 |
| TIP | TIP | USD | 동일 | 1,592 |
| LQD | LQD | USD | 동일 | 1,592 |
| GLD | GLD | USD | 동일 | 1,592 |
| DBC | DBC | USD | 동일 | 1,592 |
| VNQ | VNQ | USD | 동일 | 1,592 |
| KODEX_200TR | 278530.KS | KRW | 2019-09-02~2025-12-30 | 1,528 |
| KTB10Y | 148070.KS | KRW | 동일 | 1,528 |

부모 snapshot은 Yahoo 보관 원천이며 KTB10Y와 보조 환율 USD_KRW가 `failed`, 나머지 투자자산 11종이 `unverified`다. 선택 snapshot의 투자자산 12종과 독립 FX 2종은 모두 `unverified`다. 미국 10종과 KODEX_200TR의 선택 원천은 `yahoo_archived_close`, KTB10Y는 `yahoo_krx_low_correction`으로 구별한다.

KTB10Y의 2020-03-10을 실제 읽어 비교했다. 부모는 Open=127475, High=127500, Low=127500, Close=127500으로 Open < Low다. 기존 연구 선택본의 Low는 126795이고 나머지 OHLC·Volume은 같다. 선택본에는 과거 별도 원천 버전에서 공식 KRX 후보로 Low만 정정했다는 이력이 있다. 이번 작업은 그 행을 자동 보정하지 않았고, 공식 외부 증빙을 재인증하지 않았다. 부모의 `failed` 상태와 원본은 보존한다.

국내 ETF의 관측 누락은 참조 일정·동일 시장 자료와의 차이로 기록한다. 보존된 연구 보고서는 공식 후보 26일×2종=52행을 가격 기준 불일치 때문에 Yahoo 원천에 접합하지 않았다고 설명한다. 실제 날짜별 제외 CSV에서는 모델 공통 평가일 중 두 국내 ETF 관측 누락 24일을 확인했다. 전 기간 공식 KRX 현물 일정·상장/정지 이력은 이번에 인증하지 않았으므로 모든 차이를 공급자 누락 또는 실제 거래일 오류로 단정하지 않는다.

부모 가격은 Close와 Adj Close를 별도로 보존한다. 선택본의 설명은 provider-reported Close를 추가 조정하지 않고 사용하며 현금 분배금을 제외한 가격수익률이라는 것이다. 분할·분배금·과거 수정주가 빈티지의 독립 검증이 부족하므로 학습 승인된 총수익률로 해석하지 않는다.

## SMBS와 ALFRED는 별도 시나리오

| 항목 | SMBS | ALFRED |
|---|---|---|
| 원천 정의 | 국내 MAR 매매기준율 | DEXKOUS 뉴욕 정오 fixing의 주별 빈티지 |
| 선언 단위·방향 | KRW per USD, USD/KRW | KRW per USD, USD/KRW |
| 실제 CSV 행 수 | 1,561 | 1,599 |
| 수치/결측 | 1,561 / 0 | 1,579 / 20 |
| 중복 날짜·0 이하 값 | 0 / 0 | 0 / 0 |
| 기간 | 2019-09-02~2025-12-31 | 2019-09-03~2025-12-31 |
| 원천 actual_available_at | 전부 미확인 | 전부 미확인 |
| 연구 사용 가정 | 고시일+1달력일 10:00 KST | 수요일 빈티지의 다음날 00:00 New York |
| 최대 관측 나이 정책 | 7일 | 14일 |
| 실제 사용 최대 나이 | 160시간 | 326시간 |

행 수·결측·중복·양수 여부·사용 경과시간은 실제 CSV/Parquet에서 재계산했다. 정의와 공개 정책은 보존된 metadata 및 연구 문서의 선언이며 이번에 외부 공식 사이트에서 다시 확인한 결과가 아니다. 두 fixing을 서로 대체하거나 Yahoo KRW=X로 이름만 바꾸지 않는다. 현재 수집시각·빈티지 날짜·파일 수정시각을 실제 과거 공개시각으로 사용하지 않는다.

보존된 문서는 SMBS HTTP 전송·정정 이력, ALFRED 빈티지 일중 시각이 미확인이라고 명시한다. 일부 날짜의 현재 FRED 값과 당시 주별 빈티지 차이도 기록되어 있다. 이번에 전체 원문 HTTP/receipt 묶음을 재인증하지 않았으므로 그 과거 검증을 새로 완료했다고 주장하지 않는다.

## 실제 피처와 Walk-Forward 재계산

| 항목 | SMBS | ALFRED |
|---|---:|---:|
| features | 955 × 72 | 1,060 × 72 |
| rewards | 955 × 12 | 1,060 × 12 |
| rows | 1,453 × 51 | 1,453 × 51 |
| 실제 피처 기간 | 2020-01-02~2025-12-30 | 동일 |
| 피처 결측·Inf | 0 | 0 |
| 미확정 보상 행 | 12 | 9 |
| Validation 행 | 0 | 0 |

정렬·유일한 날짜 인덱스와 features/rewards 인덱스 일치를 직접 확인했다. 자산별 열 순서는 위 표 순서, 지표 순서는 `log_return, rsi14, macd, macd_signal, macd_hist, volatility20`다. 기본 피처는 정규화 전 값이다. `windows/`의 각 피처는 Train 전용 정규화 값이다.

| 시나리오·창 | Train/Test 행 | Train 평균 재계산 최대 오차 | population scale 재계산 최대 오차 | 정규화 피처 최대 오차 |
|---|---:|---:|---:|---:|
| SMBS 2020~2024 | 665 / 127 | 0.0 | 0.0 | 0.0 |
| SMBS 2021~2025 | 586 / 163 | 0.0 | 0.0 | 0.0 |
| ALFRED 2020~2024 | 700 / 127 | 0.0 | 0.0 | 0.0 |
| ALFRED 2021~2025 | 621 / 233 | 0.0 | 0.0 | 0.0 |

Train만 선택해 mean 및 std(ddof=0)를 다시 계산했고 `(raw_features - mean) / scale`를 저장된 window 피처와 비교했다. 각 창의 마지막 Train·마지막 Test 보상 12열이 모두 NaN인 것도 확인했다. Validation은 튜닝 전에 Train 안에서 별도 분리해야 한다.

rows의 actual/assumed 시각 중 evaluation_at 이후인 셀은 0개였다. 그러나 actual_available_at은 자산과 환율 모두 미확인이다. 이 결과는 기록된 연구 가정하의 시점 일관성 확인이며 역사적 point-in-time 검증 완료를 뜻하지 않는다.

## 제외 행: 실제 날짜별 CSV 집계

아래는 모델 기간 `[2020-01-01, 2026-01-01)`의 공통 참조 평가일 1,424행만 재집계한 결과다. 전체 원천 기간의 워밍업 이전 행까지 집계한 generation-summary와 분모가 다르다. 제외 사유는 저장된 우선 사유 한 개 기준이다.

| 사유 | SMBS | ALFRED |
|---|---:|---:|
| 국내 ETF 관측 없음 | 24 | 24 |
| 직전 평가가격 없음 | 21 | 18 |
| FX 관측 나이 초과 | 3 | 0 |
| 연속 구간 재시작 워밍업 | 421 | 322 |
| 제외 합계 | 469 | 364 |
| 실제 피처 행 | 955 | 1,060 |

이 표는 저장된 eligibility CSV의 집계를 직접 확인한 것이다. 새 감사 명령은 원천과 FX 선택·피처 재계산을 별도로 대조한다. 결측 구간을 무제한 Forward Fill하거나 휴장일에 새 관측값을 만들지 않는다.

## PPO 연결 전 남은 차이

연구 lineage는 `schema_version=experimental-real-team12-v1`, `dataset_kind=experimental_real`, `validation_status=unverified`이며 strict #15 저장 스키마와 다르다. strict reader를 완화하거나 이 연구 산출물을 승인된 형식으로 위장하지 않는다.

연구 Parquet 인덱스는 timezone-naive 날짜 라벨이다. 실제 평가시각은 `rows.evaluation_at`에 따로 있다. PPO 입력은 시간대 있는 인덱스와 같은 자산 순서가 필요하므로 단순 tz_localize로 평가 의미를 임의 확정하면 안 된다. #94 담당 변환기에서 정규화 전 RSI·가격 정규화 MACD, 로그수익률의 `expm1`, 날짜·자산·계보 의미를 맞춰야 한다. 다음 구간 rewards를 입력 시점의 후행 returns로 이름만 바꾸면 안 된다.

실제 과거 공개시각, 가격 조정·총수익 기준, 공식 시장 일정, 결측 구간별 에피소드, 장 마감 후 결정의 체결 시점·가격·비용·현금 정책이 남아 있다. 따라서 현재 산출물은 연구 수치 검증이 가능하지만 바로 승인된 PPO 학습·백테스트 입력으로 사용할 준비가 끝난 상태는 아니다. #94 변환기 구현은 이번 작업 범위에 포함하지 않는다.
