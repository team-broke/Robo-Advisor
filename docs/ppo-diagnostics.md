# PPO 모델 신뢰성 진단

이 도구는 기존 체크포인트의 **추론 결과**를 관찰한다. 학습, 모델 선택, 보상 변경,
비중 강제 조정, 데이터 승인 또는 투자 성능 인증을 수행하지 않는다.
기존 `calculate_metrics`, `evaluation_env_config`, `_make_env`, `load_episodes`,
`judge_convergence`, `label_market_regimes`, `dataframe_fingerprint`를 재사용한다.
환경의 `step()`이 제공하는 목표/실제 비중과 비용을 기록하기 위해 평가 루프만 추적용으로 감싼다.

## 조사 결과와 현재 한계

2026-11-24 조사 기준 `origin/develop`은 `24a3971342ddb617d4dd5e8315fc462a9b0953bb`다.
Issue [#1](https://github.com/team-broke/Robo-Advisor/issues/1),
[#25](https://github.com/team-broke/Robo-Advisor/issues/25),
[#27](https://github.com/team-broke/Robo-Advisor/issues/27),
[#30](https://github.com/team-broke/Robo-Advisor/issues/30),
[#42](https://github.com/team-broke/Robo-Advisor/issues/42)와 다음 열린 PR을 검토했다.
다른 PR 코드를 복사하거나 병합하지 않았다.

| 관련 작업 | 현재 진단과의 경계 |
| --- | --- |
| [#113 벤치마크 감사](https://github.com/team-broke/Robo-Advisor/pull/113) | 미병합 원천 검증을 완료했다고 간주하지 않음 |
| [#119 재학습](https://github.com/team-broke/Robo-Advisor/pull/119) | 실제 manifest·체크포인트 인계 후 사용; PR의 연구 CSV를 native 입력으로 간주하지 않음 |
| [#122 MVO](https://github.com/team-broke/Robo-Advisor/pull/122) | 비용·통화·수익률 전이 구간을 확인한 외부 비교 수익률만 입력 |
| [#124 위험 피처](https://github.com/team-broke/Robo-Advisor/pull/124) | 기존 RiskTag.available_at 계약만 사용; 새 위험 점수 생성 안 함 |
| [#125 평가 정책 초안](https://github.com/team-broke/Robo-Advisor/pull/125) | 최종 승인 전; 사용자가 평가 버전을 명시하며 기본 정책으로 채택하지 않음 |
| [#129 ANOVA 표본](https://github.com/team-broke/Robo-Advisor/pull/129) | ANOVA 계산/표본 생성과 독립 |

로컬 원래 저장소의 `artifacts/data-validation/`에서 기존 Issue #15·원천 증빙 감사 JSON
6개와 테스트 로그를 확인했다. `models/`와 `artifacts/training/`의 PPO 실행 산출물은 찾지 못했다.
로컬 연구 자료의 금융 데이터 ZIP 7개에서도 학습 모델·Monitor·학습 manifest 후보를 찾지 못했다.
이것은 조사한 위치의 현황이며 다른 팀원의 PC/비공개 저장소에 모델이 없다는 뜻은 아니다.
기존 감사의 부모 Yahoo USD/KRW·KTB10Y 오류 및 SMBS/ALFRED 공개시각 미확인 문제를
이 진단에서 보정하거나 승인하지 않는다. `training_approved=false`를 유지한다.

따라서 실제 Train/Validation/Test 성과, 자산 집중 여부, 학습 종료율은 **미확인**이다.
단위 테스트의 합성 12자산 자료·상수 정책은 기능 검증용이며 실제 PPO 성능이 아니다.

## 입력 및 재현 명령

Python 3.10과 현재 저장소의 고정 의존성을 사용한다. 신규 의존성은 없다.
명세 경로는 별도 지정하며, 명세 안의 모든 입력 경로는 `--root` 아래 상대 경로다.
모든 입력 SHA-256은 사용자가 사전 확인한 값을 고정해야 한다. 파일을 읽을 때 재검증하고
**검증한 동일 바이트**로 Parquet/JSON/체크포인트를 읽는다. 기존 파일을 변경하지 않는다.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m robo_advisor.diagnostics.inputs diagnostic-spec.json `
  --root research-inputs --output artifacts/diagnostics/new-run
```

새 출력 디렉터리만 허용한다. 이미 존재하면 실패하며 결과를 덮어쓰지 않는다.
SB3 체크포인트는 팀이 신뢰하는 파일이어야 한다. 실행은 `PPO.load(..., device="cpu")` 및
결정론적 `predict`뿐이며 `.learn()`이나 GPU 학습을 호출하지 않는다.

명세 예시의 `SHA256_*`는 자리표시자다. 실제 파일이 없으면 명령이 실패하는 것이 정상이다.
아래 날짜도 실제 실험 결과가 아닌 **형식 예시**다. native manifest와 반드시 맞아야 한다.

```json
{
  "schema_version": 1,
  "kind": "checkpoint_inference",
  "policy_version": "team-approved-version-required",
  "currency": "KRW",
  "training_manifest": {"path": "run/manifest.json", "sha256": "SHA256_MANIFEST"},
  "checkpoint": {"path": "run/best_model.zip", "sha256": "SHA256_MODEL", "run_id": "run-identifier"},
  "dataset": {
    "returns.parquet": {"path": "dataset/returns.parquet", "sha256": "SHA256_RETURNS"},
    "rsi.parquet": {"path": "dataset/rsi.parquet", "sha256": "SHA256_RSI"},
    "macd.parquet": {"path": "dataset/macd.parquet", "sha256": "SHA256_MACD"}
  },
  "source_audit": {"path": "audit.json", "sha256": "SHA256_AUDIT"},
  "periods": [
    {"name": "train", "start": "2020-01-01T00:00:00Z", "end": "2023-01-01T00:00:00Z"},
    {"name": "validation", "start": "2023-01-01T00:00:00Z", "end": "2024-01-01T00:00:00Z"},
    {"name": "test", "start": "2024-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"}
  ],
  "monitor_files": [{"path": "run/monitor/env-000.monitor.csv", "sha256": "SHA256_MONITOR"}],
  "sensitivity": true
}
```

`dataset`은 native `returns.parquet`(**단순수익률**), `rsi.parquet`(0~100),
`macd.parquet`(가격 정규화 MACD 선)이며, 연구용72피처 `features.parquet`와 다르다.
세 파일은 같은 timezone-aware 날짜와 자산 열 순서를 사용한다. 수익률은 유한해야 한다.
RSI와 MACD는 기존 환경과 같이 각 롤아웃에서 관측하는 행부터 유한 값을 요구한다.
초기 관측에 사용되지 않는 워밍업 지표는 NaN을 허용하며 보간하지 않는다.
현금은 입력 자산 열에 넣지 않으며 환경이 마지막 비중으로 추가한다. 자산 수는 하드코딩하지 않는다.
선택적 `risk_tags.parquet`는 native RiskTag 필드와 manifest의 파일 목록/해시를 따라야 한다.
로그수익률을 자동 변환하거나 정규화기를 fit하지 않는다.

완료 상태 native 학습 manifest의 seed/config SHA-256, 환경·보상·비용 설정, 데이터 파일 해시,
자산 순서, 실제 Train 시작/끝과 Validation 시작을 검사한다.
로드한 체크포인트 자체의 seed와 관측·행동 차원도 대조한다.
Test는 학습 및 모델 선택 inclusive cutoff 이후여야 한다.
Validation의 반개구간 종료 경계는 inclusive cutoff까지 기록된 모든 입력 행을 포함해야 한다.
Validation 기록이 없는 실행에 임의의 검증 구간을 만들어 붙이지 않고 오류로 처리한다.
native manifest에는 체크포인트 SHA-256 등록이 없으므로 모델과 학습 이력의 연결은
**operator_pinned_unverified**다. 같은 run_id를 적는 것만으로 모델 계보가 입증되지는 않는다.
해시는 파일 무결성 검사이며 원천 공개시각/정규화/금융 신뢰성을 증명하지 않는다.
결과 manifest의 `inputs.dataset_inputs`에 원본 데이터 파일별 상대 경로와 확인한 SHA-256이 남는다.

`kind`는 생략 시 `checkpoint_inference`다. 합성 입력이나 테스트용 체크포인트는
`kind: "synthetic"`으로 명시한다. `checkpoint_inference`는 실행 방식이며 원천 검증이나
학습 승인을 의미하지 않는다. 결과 JSON·보고서는 자료 구분을, 각 CSV는 `result_kind`를 기록한다.

### 비교 전략과 국면 입력

선택적 `references`는 다음 항목의 목록이다. 비교 원천도 반드시 해시를 고정한다.
`file`의 Parquet 열 순서는 `return`, `return_start`이며 인덱스는 수익률 전이 종료일이다.
`return_start`는 해당 수익률 시작 관측 시각이다. 전이 구간, 전체 날짜, 통화, 수수료,
슬리피지, `cost_basis="net_same_accounting"`가 다르면 명시적으로 실패한다.
입력 계약의 비용 동등성 선언은 실제 거래 로그 감사를 대체하지 않는다.

```json
{
  "period": "test",
  "name": "equal_weight",
  "file": {"path": "references/test-equal.parquet", "sha256": "SHA256_REFERENCE"},
  "commission_rate": 0.00015,
  "slippage_rate": 0.0005,
  "currency": "KRW",
  "cost_basis": "net_same_accounting"
}
```

`name`은 `equal_weight` 또는 `benchmark`다. 참조 수익률이 없으면 `unavailable`로 표시한다.
동일가중의 재조정 주기나 50:50 합성 벤치마크를 임의로 정의하지 않는다.
조기 종료나 문맥 부족이 있으면 `not_comparable_incomplete`, 초과수익률 null이다.
구간별 CSV 지표는 실제 관측된 부분의 기술 통계이며 완전한 평가기간 성과로 주장하면 안 된다.

국면별 평균 비중은 선택적 `regime_benchmark`（`return` 한 열 Parquet의 path/sha256）와
`regime_rules`를 함께 제공할 때 기존 인과적 라벨러로 산출한다. 예시 규칙은
`{"block_anchor":"2020-01-02T00:00:00Z","block_size":21,"min_blocks_per_regime":2,"lookback":60,"threshold":0.05}`다.
기본 평가 규칙으로 채택하지 않으며 팀이 고정한 값을 사용해야 한다.
라벨은 블록 시작 전의 수익률만 참조하고 투자 **결정 시각**에 연결한다.
벤치마크의 실제 공개시각과 원천 검증은 별도 감사 대상이다.
국면 입력이나 과거 문맥이 없으면 `unknown`을 유지한다.

## 결과 해석

| 산출물 | 내용 |
| --- | --- |
| summary.json | 모델/데이터 해시 출처, seed, 학습 설정, 구간 완주/비교 상태와 한계 |
| period_metrics.csv | Train/Validation/Test × 설정별 누적수익률·Sharpe·MDD·실제 날짜 |
| allocation_history.csv | 종료일/결정일, 목표/실제 위험자산+현금, 비용, 회전율, HHI, 종료 사유 |
| allocation_summary.csv | 자산/구간/국면별 평균 비중, HHI, 최대 비중, 목표 변화량, 비용 |
| safeguard_events.csv | 추론 종료 날짜/NAV/MDD와 학습 Monitor 종료 이벤트 |
| diagnostic_report.md | 한국어 성과·비교 가능성·해석 한계 보고서 |
| manifest.json | 입력 명세/해시, 설정/평가 버전, 출력 파일 SHA-256; 자기 자신의 해시는 제외 |

기간은 수익률 전이 **종료일** 기준 반개구간 `[start,end)`다. Validation/Test의 첫 전이는
직전 lookback 관측만 문맥으로 사용한다. Train은 native loader의 시작일부터 계산하므로
초기 lookback개의 수익률은 평가할 수 없어 명시적 문맥 부족으로 남긴다.
각 구간은 NAV=1, 현금=1로 독립 초기화한다. 연속 보유한 실거래 포트폴리오가 아니다.
현재 `rl.validation.split_validation` 및 PR #119의 설명은 lookback-1행 문맥으로 첫 Validation/Test
날짜를 초기 관측일로 사용해 첫 수익률이 다음 날짜부터 시작한다. 이 진단은 수익률 종료일 계약에
맞춰 lookback행의 과거 문맥으로 첫날 수익률까지 재평가한다. 따라서 native `validation.best/history`
점수를 직접 재현한 값이 아니며 같은 지표로 대조하면 안 된다. 기존 모듈은 변경하지 않았다.
첫날 수익률 포함 규칙은 Issue #1에서 합의하고 비교 입력의 실제 전이 날짜를 확인해야 한다.
미래 피처를 만드는 upstream 과정의 무누수를 이 롤아웃 자체로 인증하지 않는다.
완주 판정은 `coverage_basis=input_index`로 표시한다. 원천 거래일 누락은 별도 감사가 필요하다.
입력 시간대는 명시되어야 하며 같은 시각은 UTC로 정규화해 비교한다.
지표는 기존 `calculate_metrics`의 연252일/연무위험0/표본표준편차 정의를 사용한다.
정의 불가능한 Sharpe는 JSON null/CSV 빈 값이다.
학습 시 라이브러리 버전과 현재 추론 버전도 별도 기록한다.
Python API `compare_experiment_settings(left, right)`는 두 `DiagnosticResult`의 데이터 값/파일 해시,
평가 버전·기간·비용·보상·seed·하이퍼파라미터 차이를 나열하며 성과의 우열을 계산하지 않는다.

목표 비중은 softmax 직후, 실제 비중은 거래비용 차감 및 다음 수익률 반영 후 drift 비중이다.
비용은 초기 NAV=1 단위 금액이다. 회전율 정의도 기존 회계 함수를 따른다.
`allocation_summary.csv`의 비용·회전율·HHI는 그룹 전체 값으로 각 자산 행에 반복된다.
자산별 비용 배분이 아니므로 자산 행을 합산하면 안 된다.
HHI는 **현금을 포함**하며 최대 위험자산 비중과 현금 포함 최대 비중을 분리한다.
위험 12개 각1/12·현금0 기준과 전체13개 각1/13 기준의 목표 L1 거리를 별도로 기록한다.
시간상 정체는 목표 변화량, 집중은 최대 비중/HHI로 관찰한다. 과도함의 임계값은 정하지 않는다.

기존 Safe-Guard는 `nav < peak_nav * (1-limit)`일 때 종료한다. **15%와 같으면 종료하지 않는다.**
자동 매도/현금화가 아니다. 종료 이후 날짜는 수익률0으로 채우지 않는다.
마지막 평가 관측에서 한도를 넘기면 Safe-Guard 이벤트를 기록한다. 관측 누락이 없으면
구간은 완주로 표시하며 해당 이벤트를 조기 종료로 세지 않는다.
15% 도달/초과 날짜도 반올림된 낙폭 대신 같은 NAV 경계로 계산하므로 민감도 모드 간 경계 판정이 같다.
민감도 비교는 동일 모델·데이터·비용으로 `mdd_limit=1.0`만 적용한 추론이다.
NAV가 양수인 기존 환경에서 조기 종료를 방지하는 설정이며 학습 정책/기본값을 바꾸지 않는다.
이 결과를 Safe-Guard 없이 재학습한 실험으로 해석해서는 안 된다.

학습 Monitor는 종료 사유/보상/길이만 기록한다. 알 수 없는 사유가 있으면 종료율은 null,
기록이 없으면 0회가 아니라 `unavailable`다. 날짜/NAV/MDD/급락기 대 일반시장 빈도는
복원하지 않는다. 상세 학습 이벤트가 필요하면 담당자와 별도 계측 실험을 협의해야 한다.
종료율의 분모는 Monitor에 기록된 완료 에피소드 수다. 학습 종료 시 진행 중이던 에피소드 수는
이 로그만으로 알 수 없으며 전체 시작 횟수 대비 종료율로 해석하면 안 된다.
통합 진단에 Monitor를 제공하면 manifest/config의 n_envs와 native `env-000.monitor.csv`부터
전체 병렬 환경 파일 목록이 일치해야 한다. 일부 환경만으로 전체 종료율을 계산하지 않는다.
모든 로그를 확보하지 못했다면 monitor_files를 생략해 학습 종료 통계를 unavailable로 남긴다.
위 JSON 예시의 Monitor 1개는 n_envs=1일 때만 완전하며 실제 n_envs에 맞춰 전부 나열해야 한다.
과적합·학습 부족·보상 오류·원천 오류 중 하나를 단정하거나 Test로 설정을 탐색하지 않는다.

## 검증과 후속 작업

```powershell
python -m pytest tests/test_ppo_diagnostics.py -v
python -m pytest -q -m leakage --strict-markers
python -m pytest -v
python -m black --check src tests dashboard
python -m flake8 src tests dashboard
```

테스트는 합성 자료/정책을 명시한다. 실제 모델이 없으므로 현실의 성능 수치는 만들지 않는다.
직렬화 계약 테스트 한 개는 임시 경로에 미학습 PPO 정책을 저장하고 CPU에서 읽는다.
`.learn()`은 호출하지 않으며 실제 금융 모델의 학습·성능 검증이 아니다.
Windows의 Bash/CUDA 설치 스크립트 테스트와 `os.kill(pid,0)` 기반 생존 확인 테스트는
Linux CI와 운영체제 동작이 달라 별도로 구분해 결과를 보고한다. 기존 테스트를 수정하지 않는다.
Windows 분리 검증 명령(전체 통과 명령이 아님):

```powershell
python -m pytest -q --ignore=tests/test_cuda_torch_install.py `
  -k "not test_pid_alive_checks_only_valid_process_ids and not test_live_lock_blocks_and_stale_lock_is_taken_over"
python -m pytest tests/test_cuda_torch_install.py -q
```

후속 인계에는 신뢰된 checkpoint, 완료 manifest, 원본 전체 native 입력 파일과 고정 해시,
Monitor, 원천 감사, 동일 날짜/통화/비용의 비교 전략 수익률이 필요하다.
Issue #1에서는 Validation/배당 PR·TR/환율 시각/벤치마크 재조정/비용 및 비교 시 Safe-Guard 정책,
집중/정체의 경보 임계값과 평가 버전을 합의해야 한다.
PR #119의 5보상×3seed 구성과 #125 초안의 3보상×5seed 구성도 서로 다르므로 실험 전에 합의한다.
실제 학습 종료 국면별 통계와 과적합의 인과적 판정에는 추가 기록 또는 사전 고정한 후속 실험이 필요하다.

## 최초 구현 당시 로컬 검증 결과 (2026-11-24)

Python 3.10.21에서 실행했다. 실제 금융 학습 결과는 검증하지 않았다.

| 검사 | 결과 |
| --- | --- |
| 신규 진단 단위 테스트 | 72 passed |
| leakage / strict-markers | 92 passed |
| Black / flake8 | 통과 / 통과 |
| Windows 제약을 분리한 회귀 | 1,596 passed, 2 deselected, 24 warnings |
| CUDA 설치 스크립트 모듈 별도 실행 | 4 passed, 19 failed; WSL /bin/bash 부재 |

분리 회귀는 CUDA 설치 스크립트 모듈 23개를 제외하고 PID 생존 확인 테스트 2개를 deselect했다.
전체 `pytest -v` 실행은 기존 `os.kill(pid,0)` 생존 확인 테스트 부근에서 프로세스가 종료되어
완주하지 못했다. 전체 통과를 주장하지 않으며 Linux CI 확인이 남아 있다.
24개 경고는 기존 Gymnasium 6개와 SHAP deprecation 18개다. 기존 테스트를 수정하거나 우회해
정상 결과로 기록하지 않았다. 분리 실행의 범위와 원래 전체 실행의 한계를 함께 기록한다.

## Draft PR 전 감사 결과

- 진단·평가·Validation 선택·환경·Safe-Guard 테스트 205개 통과. 기존 Gymnasium 경고 6개.
- leakage 테스트 93개, Black, flake8 통과.
- 실제 체크포인트와 학습 당시 입력의 연결, 실제 거래일 커버리지, Linux 전체 CI는 미검증이다.
