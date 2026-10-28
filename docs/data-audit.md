# Issue #15 실제 원천·FX·피처 감사 실행

이 감사는 #9 수집기, #10 검증 규칙, #11 snapshot reader, #12 거래일 계약,
#13 지표, #14 Train scaler, #15 artifact reader를 재사용한다. 공급자 다운로드,
가격 보정, 새 금융 관측 생성, 학습 승인 및 #94 변환기는 실행하지 않는다.

## 한 번의 명령으로 실행

저장소 루트에서 Python 3.10 환경과 기존 잠금 파일의 의존성을 설치한다.
패키지를 아직 설치하지 않았다면 `python -m pip install --no-deps -e .`를 실행한다.
개발 중에는 PowerShell의 `$env:PYTHONPATH = (Resolve-Path src).Path`도 가능하다.
이 작업은 requirements/pyproject의 버전을 변경하지 않는다.

```powershell
python -m robo_advisor.data.audit --config config/team12_audit.json --root '<evidence-root>' --output '<새 보고서 경로>.json'
```

`<evidence-root>`는 보존한 `team12-experimental-<delivery-id>` 패키지의 루트다.
원천은 저장소 밖에 그대로 둔다. 출력 파일의 부모 폴더는 먼저 준비하며,
기존 출력 파일은 덮어쓰지 않는다. JSON을 stdout에도 출력하므로 파일 저장만
필요하면 PowerShell에서 `> $null`로 화면 출력을 생략할 수 있다.

`config/team12_audit.json`은 그 패키지의 상대 경로와 고정 SHA-256을 담는다.
원천 내용, API 키, 개인 절대 경로는 포함하지 않는다. 고정 해시는 기존 보존
보고서의 값과 직접 재계산 값을 대조했다. 파일이 바뀌었다고 자동으로 해시를
갱신하지 않는다. 별도 원천 버전은 독립 검토 후 새 설정으로 감사해야 한다.

종료 코드는 `0=검사 통과`, `1=결함/입력 오류`, `2=미확인`이다. 어떤 코드도
학습 승인을 뜻하지 않으며 모든 결과는 `training_approved=false`다. 현재 부모
스냅샷의 KTB10Y 및 Yahoo 환율 결함을 포함한 실제 감사는 **종료 코드 1**이 정상적인
결함 검출 결과다. 이 결과를 성공 코드로 바꾸거나 테스트 실패로 혼동하지 않는다.

## API와 보고서

- `data.audit.audit_fx_data(observations, evaluation_at, *, source, direction, unit, max_age_days, selected=None)`:
  실제 FX 행에 대한 as-of 선택을 재생하고 저장된 포인터·값·시각·나이를 대조한다.
- `data.audit.audit_research_package(config, *, root)`:
  두 snapshot, 참조 캘린더, 두 FX 시나리오, 연구 산출물과 제외 CSV를 함께 감사한다.
- `features.audit.audit_feature_artifact(path, *, expected_lineage_sha256, prices_krw=None)`:
  artifact 해시·열·시점·지표·보상·분할별 Train scaler를 검사한다. 전체 원화 가격을
  전달하면 원천에서 RSI/MACD/변동성/로그수익률 및 다음 구간 보상을 재계산한다.

보고서는 `status`, `training_approved`, `issues`, `metrics`를 제공한다.
`metrics.components`의 `parent/selected`는 원천별 기간·통화·조정 기준·검증 상태,
숫자/결측 검사와 manifest 무결성 결과를 담는다. `smbs/alfred`에는 각각 FX 원행
포인터, 최대 나이, 제외 날짜·사유, 72열 피처의 누락 비율과 window별 검사가 있다.
금융 원천 상태와 `integrity`, `selection_check`, `source_price_recomputation`을
구분한다. 수치 재계산 성공만으로 원천 상태를 `passed`로 바꾸지 않는다.

## 시점·계보·스키마 경계

SMBS와 ALFRED는 독립 원천이다. 입력 방향·단위 쌍을 필수로 검사하고 출력 환율을
KRW per USD로 명시한다. 역방향은 명시적으로 설정한 경우에만 역수 변환한다.
`actual_available_at`이 있으면 연구 가정보다 우선한다. 실제 공개가 평가 이후라면
더 이른 `assumed_usable_at`으로 허용하지 않는다. 고시/관측일, 기준 fixing 시각,
수집시각과 실제 공개시각은 서로 다르다. 결측은 그대로 세고 사용 불가 관측은
보간하거나 미래 값으로 채우지 않는다.

SMBS의 최대 7일과 ALFRED의 최대 14일은 기존 연구 시나리오의 서로 다른 정책이다.
ALFRED의 14일 허용은 정식 #12의 기본 7일 정책을 변경하지 않는다. 실제 역사적
공개 증거가 없는 시점은 미확인으로 기록한다. unit/direction 선언과 레벨 검사만으로
공급자의 정의를 독립적으로 인증하지 않는다.

실제 실험 자료는 `experimental-real-team12-v1`이며 정식 #15와 스키마가 다르다.
실험용 histogram 이름은 `macd_hist`, 정식은 `macd_histogram`이고, 실험 rows는
피처보다 긴 전체 평가축이다. 감사기는 명시적인 읽기 전용 adapter와 기존
`read_frame`의 SHA/fingerprint 검사를 사용한다. 정식 artifact에는 기존 엄격
`read_feature_dataset`을 사용한다. 어느 reader의 승인 조건도 완화하지 않는다.

원천 manifest → 원천 Close/FX → 원관측일 → 전체 미국 세션 1세션 lag → 평가시각
→ 가격 → 지표를 대조한다. rows의 FX 값뿐 아니라 관측일·원행 번호·실제/가정 시각·
기준 시각·나이·선택 상태도 별도 선택 CSV와 대조한다. 같은 환율 값이 반복돼도
잘못된 포인터를 숨길 수 없다. 제외 CSV의 True 날짜 집합과 artifact eligibility도
일치해야 한다. 제외 사유 집계는 **보존된 CSV의 사유**이며 원천 공급자의 누락
원인까지 독립적으로 확정한 결과는 아니다.

12자산 × 6피처의 고정 순서, segment별 33행 워밍업 재시작, 다음 구간 target 분리,
gap/마지막 행/분할 경계의 보상 NaN을 검사한다. 각 window의 원척도 Train만으로
새 scaler를 fit하여 저장된 평균·scale·변환값과 대조한다. 평가 구간을 fit하지 않는다.

## 실제 자료와 남은 작업

실제 원천·행 수·고정 해시·제외 집계는 [실측 인벤토리](real-data-inventory.md)에
기록했다. PPO 단위·날짜·분할·학습 승인 차이는 [#94 연결 준비](ppo-data-readiness.md)에
기록했다. 연구 자료의 파일 보존과 수치 일관성은 검사할 수 있지만 과거 공급자 공개,
전 기간 공식 현물 일정, 배당/분할/수정주가 빈티지, 실제 체결·환전·비용은 미검증이다.

기존 연구 생성 코드는 저장소 밖 패키지에 있다. 이번 작업은 그 생성기를 복사하거나
정식 빌더를 완화하지 않고 이미 보존된 산출물을 재감사한다. 깨끗한 클론에서 실제
자료를 다시 생성하려면 해당 보존 원천·연구 생성 패키지 및 명시적인 연구 정책이
필요하다. 없는 원천을 합성 데이터로 대체해서 재현 성공이라고 기록하지 않는다.

## 자동 검증

```text
python -m black --check src tests dashboard
python -m flake8 src tests dashboard
python -m pytest tests/test_market_data_audit.py tests/test_feature_audit.py tests/test_data_audit_io.py -v
python -m pytest -m leakage -v --strict-markers
python -m pytest -v
```

Windows 임시폴더 권한이 제한된 경우 새 사용자 임시폴더를 `--basetemp`로 명시한다.
신규 테스트는 작은 합성 fixture이며 외부 socket 접속을 차단한다. 실제 자료 감사와
합성 테스트 통과를 같은 검증 주장으로 합치지 않는다.

## 2026-10-27 로컬 재검증 결과

Python 3.10.21, 기준 develop 커밋 `f37443950203dbc9e1ea057c30e9af0bbd088487`에서
`feat/15-real-data-validation`으로 분리하여 실행했다. 기존 연구 브랜치의 독립
커밋과 Documents 클론의 미커밋 #14 파일은 포함하지 않았다.

| 검사 | 결과 |
|---|---|
| Black 전체 | 통과, 86개 Python 파일 |
| flake8 전체 | 통과 |
| 신규 FX 테스트 | 40 passed |
| 신규 피처 감사 테스트 | 27 passed |
| 신규 보고서·계보 테스트 | 35 passed |
| 전체 pytest | 1,191 passed, 0 failed, 6 warnings |
| leakage 선택 실행 | 63 passed, 1,128 deselected |
| 신규 파일 Python 3.10 문법 | 5개 통과 |

PR 제출 전 리뷰에서 snapshot 외부 SHA 누락과 잘못된 descriptor를 명시적 실패로
처리하도록 보완했다. 또한 미래·길이 0인 수익 구간과 정식 관측 가능 마스크의
누락·자료형·내용 불일치를 차단하는 회귀 테스트를 추가했다.

경고 6개는 기존 Gymnasium checker 두 테스트의 action/observation Box 경고다.
최초 테스트 수집 시 누락된 환경 패키지는 저장소의 고정 버전 제약에 맞춰 설치하여
해소했다. requirements/pyproject/CI 또는 다른 팀원 코드는 변경하지 않았다.

실제 감사 결과는 로컬 ignored 경로
`artifacts/data-validation/issue15-pr-review-20261027.json`에 보존했다. 이 보고서는
원천 오류·결측 위치와 날짜별 제외 행을 포함하며 금융 원문 자체는 복사하지 않는다.

- 부모 snapshot: 무결성 통과, ETF 11종 unverified, KTB10Y 및 Yahoo USD_KRW failed.
- 선택 snapshot: 무결성 통과, 14개 원천 모두 unverified.
- SMBS: 955×72, 피처 결측 0, FX 1,561행/결측 0, 최대 사용 나이 160시간,
  모델 평가일 1,424개 중 469개 제외.
- ALFRED: 1,060×72, 피처 결측 0, FX 1,599행/결측 20, 최대 사용 나이 326시간,
  모델 평가일 1,424개 중 364개 제외.
- 두 산출물 모두 해시/계보/지표 원천 재계산/각 Train scaler/분할 보상 마스크 검사에서
  오류 없음. 실제 공개·공식 일정·조정 이력은 미검증이므로 둘 다 unverified.
- 기존 strict readiness는 여전히 blocked다. 현재 38개 사유에는 이 호출에서 실제
  alignment_evidence를 제공하지 않은 정책 필드도 포함된다. 독립적인 38개 금융
  결함이 발견됐다는 뜻은 아니다.
- 실제 자료 감사의 종합 failed/종료 코드 1은 부모 원천의 알려진 결함을 보존한 결과다.
  테스트 실행 실패나 새로운 가격 보정으로 해석하지 않는다.

새로 추가한 파일은 다음 9개이며 기존 추적 파일 수정은 없다.

- `config/team12_audit.json`
- `src/robo_advisor/data/audit.py`
- `src/robo_advisor/features/audit.py`
- `tests/test_market_data_audit.py`
- `tests/test_feature_audit.py`
- `tests/test_data_audit_io.py`
- `docs/data-audit.md`
- `docs/real-data-inventory.md`
- `docs/ppo-data-readiness.md`
