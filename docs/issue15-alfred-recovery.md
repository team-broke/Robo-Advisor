# Issue #15 ALFRED 연구 데이터 복구 및 인수 준비

이 작업은 보존된 KRX 실제 시세를 이용해 국내 ETF 관측 24일을 복구하고, 기존 연구 정책의 효과를 재현한다. ALFRED를 주 검토 대상으로 사용하고 SMBS는 민감도 시나리오로 보존한다. 기존 원천·스냅샷에는 변경이 없으며 파생 데이터의 `dataset_kind=experimental_real`, 원천 검증 상태 `unverified`, `training_approved=false`를 유지한다.

## 복구 방식과 전후 결과

- KRX 화면에서 보존한 OHLCV와 조회 receipt를 일자별로 검증한다. 두 ETF의 한·미 공통 평가일 24일에서 48건을 추가했다. 미국 휴장일인 2020-07-03과 2023-07-04는 기존 범위를 따른다.
- KODEX_200TR의 KRX OHLCV는 네이버 대상 관측과 일치한다. KTB10Y는 전후 Yahoo 원시 종가와 조정계수, 네이버 조정가격을 비교했다. KRX OHLC를 조정계수의 대조 기준으로 표시했을 때 최대 차이는 1.7712원이었다. 계수는 근거 대조에만 사용했다. 산출물에는 KRX 원시 OHLCV를 저장하고 추가 행의 `Adj Close`는 비워 뒀다.
- Yahoo에 보고된 KTB10Y 분배금과 보존 운용사 기록의 연도별 금액 6건은 일치했다. 복구일과 겹치는 제공자 보고 기업행사는 없었다. 공식 분할 이력의 완전성, 모든 분배금 처리, 역사적 수정주가 빈티지와 공개시각은 미확인이다. 이 연구의 수익률은 현금 분배금 재투자를 제외한 시장 가격수익률이다.

| 기존 33관측 기준 | ALFRED 복구 전 → 후 | SMBS 복구 전 → 후 |
| --- | ---: | ---: |
| 직접 가격 누락일 | 24 → 0 | 24 → 0 |
| 가격 또는 환율의 현재·직전 입력 부재로 수익률 계산 불가 | 42 → 0 | 48 → 6 |
| 정합 연속 구간 | 17 → 1 | 20 → 4 |
| 최장 유효 피처 구간 | 236 → 1,424 | 210 → 896 |
| 재워밍업 제외 | 322 → 0 | 421 → 99 |
| 유효 피처 행 | 1,060 → 1,424 | 955 → 1,319 |

20관측 및 자산별 이력 후보의 비교표는 실행기에서 별도로 생성한다. 이 작업의 파생 데이터는 기존 33관측 기준으로 저장한다. 연속 구간 하나를 사용하는 PPO 구조에서 검토하고, 여러 구간을 연결하는 정책은 도입하지 않는다.

## ALFRED 최종 점검

- 평가 시작 전 2019-09-17부터 2019-12-30까지 연속 정합된 **71관측**이 있다. 초기 33관측은 2019-11-04에 끝났으며 2020-01-02 첫 평가 행에 이전 관측 문맥이 정상 반영됐다.
- 12자산 원화 로그수익률을 원천 가격과 ALFRED 환율에서 다시 계산한 결과는 저장값과 일치했다. 두 국내 ETF 각각 복구일과 전후 세션의 합집합 **58행**에서 로그수익률, RSI, MACD, Signal, Histogram을 독립 산식으로 대조했다. 최대 수치 잔차는 `3e-14` 미만이었다. 이 범위의 최대 절대 단순수익률은 KODEX 2.38%, KTB10Y 1.03%였다. 국내 두 ETF 전체에서도 절대 10% 수익률 스크린에 걸린 날은 없었다. 이 수치는 가격 이상 탐색 기준이며 배당·분할 이력 전체의 증빙을 대신하지 않는다. 기간 내 제공자가 보고한 KTB10Y 분배금 5건에서 원시·수정주가 일수익률 차이는 최대 3.42%p였다. 이 차이는 조정 기준을 혼합하면 안 되는 이유다.
- W1·W2 각 분할은 해당 공통 달력에서 연속적이고, 날짜를 가로지르는 보상은 차단된다. 저장된 scaler의 평균과 척도를 Train에서 재적합해 대조한다. Test 성과나 모델 적합성은 이번 검증 범위에 포함되지 않는다.

| 창 | Train 범위, 끝 배타 | Train / Validation / Test | 첫 Test 날짜 이전 연속 관측 | MVO 252개 충족 Test 월초 |
| --- | --- | ---: | ---: | ---: |
| W1 | 2020-01-01~2023-01-01 | 718 / 237 / 236 | 1,026 | 12/12 |
| W2 | 2021-01-01~2024-01-01 | 714 / 236 / 233 | 1,262 | 12/12 |

위 MVO 집계는 복구 연구본의 공통 달력을 사용한다. 정식 15:30 정합 이후의 표본과 체결 가능성은 다시 검사해야 한다.

## PPO 입력 계약과 최소 변경 제안

현재 `dataset.py`는 검증된 Yahoo 원천, 최대 7일 FX 및 15:30 검증 시각을 요구한다. `ppo_input.py`는 기존 dataset kind, 15:30에 맞춘 평가시각, 자산·FX의 실제 가용성 증빙을 검사한다. `ppo_preflight.py`도 native 입력 형식과 15:30 일정을 검증한다. 이번 PR에서는 이 세 계약을 유지한다. 복구 연구본의 1,424행은 16:00 1,418행과 17:00 6행으로 기록됐으며 두 국내 ETF의 연구용 당일 종가 가용 시각이 모두 15:30 이후다. ALFRED 실제 공개시각 확인 행은 0행이다. 저장된 가정 아래 관측 기준 나이가 7일보다 긴 행은 856행이고 최댓값은 326시간이다.

후속 최소 변경안은 다음 순서를 권장한다.

1. **FX 정책을 식별자로 고정한다.** ALFRED DEXKOUS 선택 시 1 USD당 KRW 단위, 뉴욕 정오 관측 기준, 빈티지·발표 근거의 SHA-256, 최대 관측 및 공개 경과시간 14일을 명시적으로 버전 관리한다. 14일은 이 연구본의 가정이다. 기존 Yahoo 7일 기본값은 그대로 두고 지원하지 않는 공급자·정책 조합은 오류로 처리한다. 14일 설정의 정식 사용은 팀 검토와 권한 확인에 따른다.
2. **시점과 증거를 분리한다.** `reference_at`, 실제 `available_at`, `assumed_usable_at`, `decision_at`, `execution_at`을 별도 필드로 유지한다. 실제 증거를 우선하고 `reference_at ≤ available_at ≤ decision_at`을 검증한다. 누락된 실제 시각을 연구 가정으로 채워 정식 `available_at` 계약에 넣는 경로는 두지 않는다. 평가·체결 선택지는 [#125 별도 제안](issue15-evaluation-timing-proposal.md)에 정리했다.
3. **계보와 native 출력 경계를 고정한다.** `experimental_real`은 외부 해시로 고정된 검토 전용 입력으로 유지한다. 근거와 정책이 승인된 뒤에만 새 정합 산출물에서 기존 strict 구조의 `returns.parquet`, `rsi.parquet`, `macd.parquet`, `manifest.json`을 생성하는 변경을 검토한다. 변환은 원천·파생·정책 해시와 재현 가능한 계산을 검증해야 한다. 타입 이름만 바꾸거나 학습 승인 상태를 상속시키는 경로는 두지 않는다.

## #138 인수 정보

다음은 검토 요청에 사용할 식별자다. 승인 기록으로 사용될 서명·발급자·승인 범위의 형식은 #138에서 정한다.

| 항목 | 값 |
| --- | --- |
| 복구 원천 스냅샷 manifest SHA-256 | `999a5edf39f5332c95a8a81d63054f2951c033233c1b590dd6451f6cc537b33b` |
| ALFRED 파생 lineage SHA-256 | `ff86571965cb21964c418c83b0ea47db2dec5f2f669a236360167d151f39fc99` |
| W1 Train 승인 검토 범위 | `2020-01-01` 이상 `2023-01-01` 미만, 서울 날짜 |
| W2 Train 승인 검토 범위 | `2021-01-01` 이상 `2024-01-01` 미만, 서울 날짜 |
| 원천 및 학습 상태 | `unverified`, `training_approved=false` |
| native PPO 파일과 해시 | 아직 없음 |

새 native 파일을 만들 수 있게 되면 승인 요청에 파일별 SHA-256, 변환 manifest 해시, 원천·파생 lineage 해시, 평가 정책의 버전·해시, 윈도우별 Train 범위와 Validation/Test 경계를 연결한다. 승인 발급 전에 preflight 검증을 실행한다. 가용성, 체결, 가격 조정 및 이용 권한의 미확인 사항을 검토 요청에 함께 제공한다. [Issue #138](https://github.com/team-broke/Robo-Advisor/issues/138)

## 원천 이용 범위와 PR 파일

KRX CSV는 Data Marketplace 웹 화면에서 내려받았다. 해당 웹사이트 약관은 사전 허락 없는 복제·배포 등을 제한한다. Open API에도 별도 제3자 제공 제한이 있다. Yahoo의 일반 서비스 재사용 조건을 확인했고 네이버 시세의 재배포 허가도 확보되지 않았다. 인가된 리뷰어는 승인된 경로에서 자료를 개별 확보하거나 재배포가 허용된 자료를 사용한다. [KRX 웹 약관](https://data.krx.co.kr/contents/MDC/INFO/informationController/MDCINFO003.cmd) · [KRX Open API 약관](https://openapi.krx.co.kr/contents/OPP/INFO/OPPINFO002.jsp) · [Yahoo 서비스 약관](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html) · [네이버 서비스 약관](https://policy.naver.com/rules/service.html)

DEXKOUS에는 `Public Domain: Citation Requested` 표기가 있다. 이 보존 빈티지는 St. Louis Fed의 ALFRED 그래프 CSV 경로로 확보했다. 현재 FRED 일반 서비스 약관에 기계학습 관련 이용 제한이 있고 API에도 별도 조항이 있다. 공개 시리즈 표기만으로 이 확보 경로의 연구 학습 허용을 확정하지 않는다. 권리자 확인이나 별도의 사용 가능한 연준 이사회 H.10 경로 검토 후, 후자의 경우 시점·값·빈티지 계약을 새로 검증한다. [DEXKOUS](https://fred.stlouisfed.org/series/DEXKOUS) · [FRED 서비스·API 약관](https://fred.stlouisfed.org/legal/terms/) · [연준 이사회 H.10](https://www.federalreserve.gov/releases/h10/) · [연준 이사회 자료 권리 안내](https://www.federalreserve.gov/disclaimer.htm)

PR에는 소스 코드, 합성 회귀 테스트, 문서와 집계 결과만 포함한다. KRX 시세 CSV, Yahoo·네이버 응답 및 receipt, ALFRED·SMBS 빈티지 원문, 복구 가격 행, 스냅샷과 파생 Parquet는 로컬 보존 대상이다. 이들의 해시와 원천 경로 형식을 재현 문서에 기록한다.

리뷰 순서는 기존 `audit_feature_artifact()`의 동작 유지, 별도 `experimental_real` 저장·읽기 계약, 실관측 복구 및 피처 재계산, 승인 차단 순이다. `dataset.py`, `ppo_input.py`, `ppo_preflight.py`와 평가 프로토콜 원문에는 변경이 없다. `tests/test_research_dataset.py`와 `tests/test_research_recovery.py`에서 표준 경로의 변환 거부, 해시 불일치, 원천 대조 오류, 미래 시점, 단절 보상을 검증한다.

## 로컬 재현

승인된 기존 아카이브, 원응답 증빙 및 단절 감사 CSV가 준비된 상태에서 저장소 루트에서 실행한다. 실행기는 새 출력 폴더에만 저장한다. 원천을 수집하거나 학습을 실행하는 명령은 포함하지 않는다.

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
$archive = '<허용된 연구 아카이브 루트>'
$vendor = 'artifacts/issue15-etf-recovery/sources'
$gap = 'artifacts/issue15-gap-audit/excluded-dates.csv'
python scripts/recover_team12_research.py --archive $archive --vendor-evidence $vendor `
    --gap-report $gap --output artifacts/issue15-etf-recovery/reproduced-33
```

입력 구조는 `config/team12_audit.json`에 정의되어 있다. 아카이브 내 `evidence/official-KRX-originals/`의 종목·일자별 CP949 CSV와 request JSON이 필요하다. 공급자 증빙 폴더에는 `278530`과 `148070` 각각의 `<코드>.KS.yahoo-chart.json`, `<코드>.KS.yahoo-receipt.json`, `<코드>.naver-day.txt`, `<코드>.naver-day-receipt.json`이 필요하다. 기존 산출물 점검에는 외부에서 전달된 스냅샷·lineage SHA-256을 사용한다. 새 실행은 snapshot ID를 다시 발급하므로 새 manifest 해시를 사용하되 원천 핀과 계산 결과를 대조한다.

```powershell
$run = 'artifacts/issue15-etf-recovery/recovered-33'
python scripts/verify_alfred_research.py --archive $archive `
    --dataset "$run/datasets/alfred-recovered-33-shared" `
    --snapshot "$run/snapshots/20261127T123138344522Z_77487ac6d81b44da9ba5c33ef379732c" `
    --snapshot-sha256 999a5edf39f5332c95a8a81d63054f2951c033233c1b590dd6451f6cc537b33b `
    --lineage-sha256 ff86571965cb21964c418c83b0ea47db2dec5f2f669a236360167d151f39fc99 `
    --recovery-evidence "$run/recovery-evidence.csv" --evidence-lineage "$run/evidence-lineage.json"
```

로컬 검증 결과와 #138 검토 요청은 각각 `artifacts/issue15-alfred-final/validation.json`과 `artifacts/issue15-alfred-final/issue138-handoff.json`에 보관한다.

최종 검토에서 관련 회귀 테스트 196개가 기본 환경에서 통과했고, Windows junction 테스트 1개는 일반 임시 폴더에서 따로 재실행해 통과했다. `pytest -m leakage` 114개와 `black --check src tests scripts dashboard`, `flake8 src tests scripts dashboard`, `git diff --check`가 통과했다.

PR 제목은 `feat: 국내 ETF 누락 관측 복구 및 연구용 데이터셋 생성 (#15)`을 제안한다. 본문은 `Refs #15`로 연결하고 정식 학습 승인은 열린 상태로 둔다.
