# 공시 Event Study 및 RiskTag 검증

이 도구는 OpenDART 원문에서 후보를 추출하고, 공시 가용 시점 이후의 시장 반응을
연구용으로 측정한다. 이벤트 후보, 수익률, 기존 PPO의 RiskTag 입력은 별개다.
기존 `risk_score = severity × confidence` 계약을 유지하며 이 도구는 점수를 발행하지
않는다. 네이버 뉴스 저장·색인과 모델 호출 없이 실행한다.

## 실행과 산출물

신뢰 SHA-256은 검증된 수집 이력이나 팀 전달 문서에서 받아 사용한다.

```powershell
$env:PYTHONPATH = 'src'
python scripts/run_dart_event_study.py `
  --corpus artifacts/live-dart-body/research_manifest.json `
  --corpus-sha256 fd7410ccf5602865a7cfb2f41813e892a546ebd71f66b142cdd243e3efd722e7 `
  --as-of '2026-11-26T00:00:00+09:00' `
  --source-kind real_opendart `
  --output artifacts/dart-event-study-reproduction
```

출력 폴더는 새 이름으로 지정한다. CLI는 다음 파일을 만든다.

| 파일 | 내용 |
|---|---|
| `event_candidates.jsonl` | 후보 유형과 기존 RiskTag 유형 매핑, 발행사 종목코드, 접수번호, 원문·문서·발췌 해시, 가용 시각, 연구 trace |
| `event_study_results.csv` | 1·3·5·10거래일 성과와 시점·데이터·중복 제외 사유 |
| `event_type_summary.csv` | 유형·분할·기간·계산법별 건수, CAR 분포, 독립 구간의 평균 및 신뢰구간 |
| `calibration_report.md` | 확인된 원문 범위, 미확인 근거, 보정 상태 |
| `manifest.json` | 고정 입력 해시, 방법 버전, 분할·방법 설정, 원문별 검토 내역, 출력 해시 |

표본을 확장하려면 상호 확인된 공시 묶음마다 `--corpus` 및
`--corpus-sha256`를 같은 순서로 추가한다. 동일 접수번호는 한 번만 집계한다.
평가 시점 안에 확인된 정정 표시나 동일 접수번호의 원문 SHA-256 충돌은
해당 원문의 반응 분석을 보류한다.

팀 독립 검토에서는 기업·종목코드·접수번호, 원문 형식과 해시, 증빙 위치,
사건일과 당시 최초 공개 근거, 정정 관계, 검토자와 판정을 로컬
`event_review.csv`에 기록한다. HTML 원문은 팀원이 원문을 별도로 대조할 수
있으며 현재 자동 분석기의 입력 범위는 XML이다. 독립 판정이 없는 표본은
점수 보정에 사용하지 않는다.

가격 분석에는 `--market`과 `--market-sha256`, `--policy`와 `--policy-sha256`가
필요하다. 가격 manifest는 `schema_version=1`, `kind=real`, `venue=KRX`,
`currency=KRW`, `training_approved=false`와 다음 CSV 기술자를 받는다.
각 기술자는 상대 `path`, `sha256`, `source_url`, `verification=verified`,
`identity`, `basis`를 가진다. 가격과 달력 파일의 독립 검토를 마친 뒤 팀에서
신뢰 해시를 전달한다. 해시는 파일의 무결성을 확인한다.

| 키 | `identity` / `basis` | CSV 필드 |
|---|---|---|
| `calendar` | `KRX` / `official_session_open_close` | `session_date,open_at,close_at` |
| `benchmark` | `KOSPI` / `price_index` | `session_date,close,available_at` |
| `issuers.<stock_code>` | 6자리 발행사 종목코드 / `split_adjusted_price_return` | `session_date,close,available_at` |

가격 기술자에는 `availability_basis` (`verified_release_timestamp` 또는
`conservative_verified_bound`)와 해당 근거의 `availability_source_url`도
기록한다. 거래 일자는 한국 현지 날짜 `YYYY-MM-DD`, 시각은 오프셋을 포함한다.
종목 가격은 기업행위로 인한 단절을 검토한 분할 조정 종가를 사용하며, 두 시계열
모두 분배금 재투자를 포함하지 않는다. 확인이 필요한 거래정지·휴장·결측을 별도
기록하고, 공식 거래일을 가격 CSV 행으로 추론하거나 가격을 채우지 않는다.
공식 원천은 KRX의 주식 일별매매정보와 지수 시계열, 거래일 자료다.
[KRX 주식 Open API](https://openapi.krx.co.kr/contents/OPP/USES/service/OPPUSES002_S2.cmd?BO_ID=JvJFzlAENzZlPBDNGAWC) ·
[KRX 정보데이터시스템](https://data.krx.co.kr/)

분할 정책 JSON에는 날짜 `start`와 `end_exclusive`가 있는 `splits` 배열을
Train·Validation·Test 순서로 지정한다. 방법 기본값은
`estimation_sessions=120`, `pre_event_gap_sessions=10`, `min_ci_events=10`이다.
팀 검토 후 정책을 고정하고 해시를 남긴다. 출처 유형은 기본값 `unverified`이며
실제 원천을 확인한 실행에만 `real_opendart`를 선언한다. 합성 자료에서는
`synthetic`을 사용한다.

## 후보의 해석과 계산 기준

- OpenDART 목록과 원문, 발췌 해시를 재현한 뒤 본문의 문단과 표 행을 조사한다.
  명시적인 실적 쇼크·악화, 계약 해지, 주요 경영 변경, 규제 변경, 급등락 또는
  지정학 표현을 선별한다. 검토 대상은 `candidate_only`다. 기존 RiskTag에
  대응하지 않는 유형의 `risk_tag_type`은 비워 둔다.
- 공시는 알려진 최신 로컬 수집 시점에서 검증하며 정확한 공개 시각 대신
  접수일 정확도와 기존 본문 가용 시각을 사용한다. 정정 연결 미확인·철회·HTML은
  본문 검증 단계에서 제외된다. 주주·계열회사 등 관련 주체의 경영·계약 언급은
  해당 발행사와의 직접 관계가 있어야 후보가 된다. 명시 날짜가 접수일보다
  모두 30일 넘게 앞선 구절은 최초 공시 여부를 확인할 필요가 있어 보류한다.
  이 기간은 후보 검토 기준이며 시장 반응이나 RiskTag 점수의 임계값이 아니다.
  발행사의 검증된 DART 종목코드만 사용하며 ETF로 전파하지 않는다.
- 첫 거래일은 `available_at` 이후 처음 개장하는 정규 거래일이며 직전 종가를
  기준으로 복리수익률과 MDD를 계산한다. 장중 가용 공시는 다음 개장부터 측정하므로
  당일 장중의 즉시 반응은 별도 자료가 필요하다. 기준 종가는 첫 개장 전에
  가용해야 한다. 기준일이나 관측 끝이 분할 경계를 벗어나거나 필수 관측이 없으면
  해당 구간을 제외한다.
- 시장조정 일별 초과수익률은 `종목 단순수익률 − KOSPI 단순수익률`이다.
  시장모형의 초과수익률은 `종목 단순수익률 − (α + β × 시장 단순수익률)`이다.
  시장모형은 공시 가용 시각보다 앞서 닫혔고 가격도 가용했던 관측에서
  OLS로 적합한다. 추정 이력이 모자라도 단순 시장조정 결과는 별도 제공한다.
  CAR는 일별 초과수익률의 합이며, 구간 복리수익률의 차와 각각 기록한다.
  실현 변동성은 비연환산 일별 표본 표준편차이므로 1일 구간은 비워 둔다.
  [MacKinlay의 Event Study 방법론](https://www.bu.edu/econ/files/2011/01/MacKinlay-1996-Event-Studies-in-Economics-and-Finance.pdf)
- 같은 공시·유형의 여러 구절은 후보 하나로 묶으며, 동일 자산·유형의 정확히
  반복된 발췌는 중복으로 제외한다. 기준일부터 종료일까지 겹치는 구간은
  종목과 무관하게 같은 클러스터로 기록하고, 기술 통계에는 표시하되 독립
  표본 신뢰구간에서 제외한다. 정규성, 장기 의존성, 동시 뉴스의 영향을 별도로
  검토하기 전에는 95% t 신뢰구간을 탐색적 결과로 해석한다.

## RiskTag 점수 검토

Severity는 공시 시점에 알 수 있는 계약 규모·실적 변화 등으로 추정할 경제적
영향의 크기다. Train에서만 사전 원문 특성과 사후 시장조정 손실·CAR의 관계를
검토할 수 있다. 후보 방법으로 이벤트 유형별 음의 CAR 분포를 Train에서
추정한 뒤 공시 시점의 특성으로 상대적 손실 크기를 예측한다.
Confidence는 독립 검토자 간 분류 일치와 원문·시각·종목 매핑의 검증 수준이다.
Validation에서는 고정 규칙의 재현성과 보정 오차를 검토하고,
Test에서는 확정 후보를 평가한다. 사후 수익률을 PPO 관측으로 내보내는 경로는
없다. LLM 자기평가와 독립 표본이 부족한 연구 결과는 점수 근거로 쓰지 않는다.

## 2026-11-26 실제 공시 점검

기존 실제 원문 세 건의 해시를 확인했다. 삼성전자 분기보고서와 계열회사
상품·용역거래 변경 공시에서 본문을 검토했고, 지분 공시 한 건은 HTML 형식으로
제외했다. 분기보고서의 주주 관련 구절 두 개에 대표이사 사임·선임 이력이
있었으나, 보고 기업 자체의 새 이벤트로 연결할 근거가 없어 후보에 포함하지
않았다. [삼성전자 분기보고서](https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20261116001482)

최종 후보와 수익률 산출 구간 모두 0건이다. 검증된 종목·KOSPI 가격과 가용
시각, 공식 거래일, 연구 분할을 이 실행에 제공받지 못했다. 추가로 직접 관련된
공시와 독립적으로 검토한 사건 라벨, 원본 및 정정 이력이 필요하다. 상태는
`insufficient_evidence`, `training_approved=false`로 남겼다.
