# 통합 백테스트 후보 정책 (#27)

이 도구는 완료된 PPO 실행 한 건과 같은 윈도우의 MVO, 월간 동일가중,
KOSPI·S&P 500(KRW) 합성 지수를 비교한다. 현재 평가는 후보 정책으로만
기록하며 성능 기준 판정은 `pending_protocol_and_evidence`로 보류한다.
결과의 `status`는 `unverified`, `training_approved`는 `false`다.

## 기존 동작과 통합 처리

| 기존 동작 | 통합 비교 |
| --- | --- |
| #119 PPO·스텝별 동일가중은 환경의 MDD 종료를 사용, #122 MVO는 끝까지 진행 | `configured`에서는 PPO 경로가 일찍 끝나면 비교를 막는다. `observe_only`는 기존 진단기의 같은 체크포인트 추론 모드를 사용한다. 학습 환경과 종료 기준은 유지한다. |
| 기존 동일가중은 softmax로 매 스텝 목표를 정하고 미세한 현금을 남김 | 별도 월간 전략은 위험자산별 1/N, 현금 0을 목표로 첫 전이와 현지 월 첫 평가 세션에 끝나는 전이의 시작점에서 리밸런싱한다. 중간은 표류한다. |
| 재학습 요약은 비교 수익률 누락 시 0 수익률을 사용 | 누락·비유한·다른 날짜에서는 Alpha, Beta, IR을 비워 둔다. 남은 기존 상대 지표의 기준은 스텝별 동일가중이며 시장 벤치마크와 구분한다. |
| #119 Test와 기존 Validation은 첫날을 초기 관측으로 사용 | 후보 Test 정책으로 첫 종료일 포함 또는 초기 관측 이후부터 선택한다. 기존 학습의 Validation 모델 선택은 그대로 유지한다. 진단의 Validation 재평가는 첫날 포함 기준이다. |

기준은 같은 KRW 단순수익률, 전이 시작·종료 **시각**, 초기 NAV 1과 현금 100%다.
세 투자 전략은 checkpoint 환경의 수수료·슬리피지와 기존 `rebalance_step` 회계를 공유한다.
기존 재학습 결과의 초기 상태 표기는 환경의 실제 현금 시작과 일치하도록 수정했다.
동일가중 및 MVO 리밸런싱에는 초기 편입 비용도 포함한다. 벤치마크는 비용 없는 참조지수다.
합성 수익률은 감사가 재현한 두 단순수익률에 일별 0.5씩 곱해 더한다. 각 원천 지수 PR/TR이
전략 입력의 명시된 기준과 일치해야 한다. 전략 원천의 배당 포함 여부와 공식 공개 시각은
별도 증빙이 필요하다. #125는 ALFRED를 본 결과, SMBS를 민감도로 지정한다.
실행마다 한 원천을 명시하고 이를 별개 후보 결과로 남긴다.

## 월간 리밸런싱과 수익률 종료일

동일가중과 MVO는 서울 시간으로 **수익률 종료일**의 월을 구분한다. 월 첫 평가일에
끝나는 수익률의 새 목표 비중은 **직전 공통 평가 세션인 전월 마지막 평가 세션의
전이 시작점**에서 장부에 적용한다. 이후 종료일의 자산 수익률을 그 비중에 반영한다.
예를 들어 평가 일정에서 1월 31일과 2월 1일이 연속이라면, 2월 1일로 기록되는 수익률은
1월 31일 평가 시각부터 2월 1일 평가 시각까지의 전이이며 비용은 1월 31일의 전이
시작 NAV에서 차감한다.

`daily_returns.csv`의 `return_start`와 `return_end`가 두 시점을 나타낸다.
동일가중과 MVO의 `rebalance_history.csv`도 해당 전이를 `return_end`로 표시한다.
두 CSV를 전략과 종료 시각으로 연결하면 비용을 어느 전이 시작점에 적용했는지 확인할 수 있다.
이 시각은 일별 수익률 비교를 위한 **회계 처리 시점**이다. 당시 주문이 가능했는지,
마감 가격으로 체결할 수 있었는지 판단하려면 실제 관측 가용 시각, 시장별 거래 시간과
주문·체결 가격의 증빙이 필요하다.

MVO는 종료일보다 엄격히 앞선 수익률로 목표 비중을 추정한다. 이 히스토리에는
직전 평가 세션에 끝난 수익률까지 들어갈 수 있다. 월 첫 평가일에 끝나는 수익률은
목표 비중 추정에 사용되지 않는다. 동일가중 목표는 위험자산 1/N으로 고정이다.
중간 세션에서는 두 전략 모두 기존 비중을 유지한 상태로 수익률이 반영돼 비중이 표류한다.

각 윈도우의 **첫 평가 전이**에서는 월 경계와 관계없이 초기 NAV 1과 현금 100%에서
포트폴리오를 편입한다. `include_first_session`은 첫 평가일의 직전 세션에서,
`after_first_observation`은 첫 평가일을 초기 관측으로 둔 뒤 두 번째 평가일에
끝나는 전이의 시작점에서 편입한다. 회계는 전이 시작 NAV에서
`회전율 × (수수료율 + 슬리피지율) × 전이 시작 NAV`를 먼저 차감한다. 남은 NAV에
목표 비중과 해당 전이 수익률을 적용하므로 첫 순수익률과 초기 NAV 1을 기준으로
계산하는 MDD에 초기 편입 비용이 포함된다.

## 입력과 실행

검증된 로컬 작업 폴더에서 아래 명령을 실행한다.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m robo_advisor.backtest.comparison --spec comparison.json --root research-inputs --output artifacts/integrated/new-run
```

명세 예시의 해시는 실제 값으로 교체한다.

```json
{
  "diagnostic": {"path": "diagnostic.json", "sha256": "SHA256_DIAGNOSTIC"},
  "schedule": {"path": "schedule.csv", "sha256": "SHA256_SCHEDULE"},
  "benchmark": {"path": "benchmark-spec.json", "sha256": "SHA256_BENCHMARK_SPEC"},
  "policy": {
    "version": "pr125-candidate",
    "first_return": "include_first_session",
    "safeguard": "observe_only",
    "return_type": "PR",
    "fx_source": "alfred",
    "mvo_lookback": 252,
    "mvo_max_weight": 0.4
  }
}
```

- `diagnostic.json`의 학습 manifest, 모델, 입력 Parquet, 구간 명세는
  [PPO 진단 입력 계약](ppo-diagnostics.md)을 따른다. 금번 CLI는 공통 벤치마크 비교를
  직접 수행하므로 기존 진단의 선택적 `references`로 대체하지 않는다.
- `schedule.csv`는 전체 Train 시작부터 Test 종료 전까지의 예정 공통 세션을 담는
  단일 `evaluation_at` 열이다. 시간대가 있는 실제 평가 시각을 제공한다. 달력 자체의
  누락·공식 여부는 별도 증빙으로 확인한다. 자산 입력이 이 일정에서 빠지거나 추가되면
  호출을 거부하며, 컨버터에서 보고한 불연속 구간을 이어 붙이지 않는다.
  벤치마크 감사가 평가 일정에서 누락된 세션을 식별하면 해당 비교도 거부한다.
- `benchmark-spec.json`은 [벤치마크 감사 입력](benchmark-data-validation.md)이다.
  첫 평가 수익률을 계산할 수 있게 직전 공통 세션을 감사 범위에 포함한다. 두 지수 또는 FX
  구간이 빠지면 벤치마크만 불완전하다. 경로의 기존 NAV·비용은 유지되며,
  일별 교집합을 만든 뒤 성과를 재계산하는 방식은 사용하지 않는다.
- `benchmark`를 생략하거나 감사가 실패하면 시장 비교는 보류한다. 세 투자 경로가
  완주한 경우 PPO의 월간 동일가중 대비 누적 차이는 별도로 남는다.
- `include_first_session`은 과거 lookback 관측으로 첫날 종료 수익률을 평가한다.
  `after_first_observation`은 첫 평가 세션을 초기 관측으로 남긴다. 필요한 과거가 없으면
  PPO 경로가 불완전하며, 전이 시작점이 없거나 달력에 구멍이 있으면 비교 입력을 거부한다.
  첫 관측을 제외하기 전에 요청된 Test 구간이 모델 선택 구간과 겹치지 않는지 검사한다.
- 각 윈도우를 별도로 실행한다. Test를 학습 입력으로 보내거나 윈도우 사이의 NAV를
  자동으로 연결하지 않는다. `observe_only`는 MDD 종료 제한을 푼 **추론** 설정이다.

## 결과와 후속 검증

- `daily_returns.csv`는 전이 시작·종료 시각, `nav_history.csv`는 초기 NAV를 포함한
  경로, `rebalance_history.csv`는 거래 회전율·비용을 담는다. 계획 세션, 실제 경로 및
  거래 가능성은 `evaluation_coverage.csv`의 행 단위 관측 플래그에서 확인한다.
- `strategy_metrics.csv`는 완주 경로의 절대 지표와 허용된 상대 지표를 담는다.
  시장 벤치마크는 한 구간이라도 빠지면 그 윈도우의 벤치마크 NAV, 시장 상대지표를
  보류한다. `safeguard_events.csv`는 각 경로의 최초 15% 도달 및 엄격한 초과 시각을
  기록한다. 실제 PPO 에피소드 종료 사유는 manifest의 진단 항목에 보존한다.
- `ppo_period_metrics.csv`, `ppo_allocation_summary.csv`, manifest의 PPO 진단 항목에는
  Train/Validation/Test, 현금·자산 비중, HHI, 균등 목표와의 거리, 각 평가 환경을 남긴다.
  `manifest.json`은 입력·출력 해시, 후보 정책, 실행·모델 식별과 데이터 상태를 담는다.
  각 CSV에도 결과 종류·후보 버전·학습 승인 상태를 기록한다.
  CLI는 비교 명세 자체의 SHA-256도 남긴다. 산출물 줄바꿈은 LF로 고정한다.
- `prepare_anova_reward_samples()`는 호출자가 지정한 reward×seed×window 설계에서
  누락/불완전 항목을 보고한다. 합성 결과와 미완주 결과는 표본에서 차단한다.
  전체 설계가 갖춰진 경우 #42의 보상 표본과 선택 보상에 대한 전략 블록 표본을
  만들 수 있다. 국면별 표본에는 승인된 블록 규칙과 테스트 전 연속 벤치마크 이력이
  필요하다. 이 도구는 p값을 계산하지 않는다.

[#125](https://github.com/team-broke/Robo-Advisor/pull/125)의 [평가 프로토콜](evaluation-protocol.md)이
병합됐다. 프로토콜은 첫날을 포함하면서 lookback−1행을 빌리도록 명시하므로,
종료일 기준 lookback행이 필요한 기존 환경과 어떻게 맞출지 확인이 필요하다.
#119의 현재 설정은 보상·λ 조합 5개×seed 3개이며 #125 본실험 설계는 보상 3종×seed 5개다.
프로토콜의 시장 벤치마크 대비 샤프비율 +0.2와 누적수익률 +10%p는
완료된 두 테스트 윈도우의 결과로 판정한다. 현재 비교 도구의 산출물은 후보 결과이며
두 기준의 달성 여부는 보류한다.

[#126](https://github.com/team-broke/Robo-Advisor/pull/126)의 사전 로더는 신뢰 SHA-256으로
고정한 `run_manifest.json`과 `files.metrics.json` descriptor를 사용해
`metrics.json`의 `records`를 읽거나, 완료된 #119 manifest와 별도 SHA-256의
`summary.csv`를 읽는다. 레코드는 strategy/window/start/end/metrics를 사용하고
중복 strategy/window를 허용하지 않는다. 본 비교 결과의 manifest·CSV는 직접 로드되지
않는다. 이후 어댑터가 승인된 실행만 선택해 레코드·해시를 생성하고,
상태·평가 시작/종료 시각·복수 seed의 표현을 API 담당자와 합의해야 한다.

## 실제 성과 상태

조사한 프로젝트 checkout과 연결된 로컬 데이터 위치에서 실제 체크포인트 및 당시의
PPO 입력을 확보하지 못했다. 합성 테스트는 임시 폴더에서 16 스텝 CPU 학습, 체크포인트
검증·로드, 원천 감사 재실행, 네 전략의 산출물 생성만 검증한다. 실제 투자 성과·학습 승인과
통계 판정은 보류한다. 필요한 자료는 동일 run_id의 모델·manifest·원본 inputs·평가 일정,
배당과 FX의 시점 증빙, 완료된 복수 윈도우·seed 결과다.

## 테스트

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m pytest -q tests/test_backtest_comparison.py tests/test_equal_weight_backtest.py
python -m pytest -q -m leakage --strict-markers
python -m black --check src tests dashboard
python -m flake8 src tests dashboard
```

현재 Windows 환경에서 전체 회귀는 세 개의 프로세스 생존 확인 테스트를 제외했다.
해당 테스트는 POSIX의 `os.kill(pid, 0)`에 의존한다. CUDA 설치 스크립트
테스트도 별도 실행했으며, WSL의 `/bin/bash` 부재로 19개가 실패하고 4개가 통과했다.
이 항목을 포함한 전체 검증은 Linux CI에서 확인한다.

최초 구현 검증에서 플랫폼 제약을 분리한 전체 회귀 1,708개가 통과했다.
최종 감사 후 관련 회귀 182개와 leakage 104개가 통과했다. 관련 회귀에서는
Windows PID 테스트 1개를 제외했다. Black과 flake8도 통과했다.
