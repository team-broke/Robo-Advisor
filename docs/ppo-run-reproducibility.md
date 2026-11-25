# PPO 학습 사전 검사와 실행 전달

사전 검사는 학습을 시작하기 전에 변환된 PPO 입력, 평가 일정, 학습 설정을 대조한다.
전달 도구는 완료된 실행의 원본 자료를 새 디렉터리에 복사하고 기존 PPO 진단기용
`diagnostic.json`을 만든다. Issue #15의 원천 검증과 Issue #61의 운영 연동이 후속이다.

## 실행

설치한 저장소에서 다음 명령을 사용한다. `ROOT`는 명세에 기재된 입력 파일을 포함하는
공통 상위 디렉터리다. `BUNDLE`과 진단 출력 디렉터리는 새 경로를 지정한다.

```bash
python -m robo_advisor.ops.ppo_reproducibility preflight --root ROOT --spec preflight.json
python -m robo_advisor.ops.ppo_reproducibility pack --root ROOT --spec completed-run.json --output BUNDLE
python -m robo_advisor.ops.ppo_reproducibility verify --package BUNDLE
python -m robo_advisor.ops.ppo_reproducibility diagnose --package BUNDLE --manifest-sha256 HASH --output DIAGNOSTICS
```

JSON에 입력 설명자를 `{"path":"data/returns.parquet","sha256":"<64-character lowercase digest>"}`
형태로 기록한다. `path`는 `ROOT` 기준 슬래시 구분 상대 경로다. SHA-256은 팀에서 신뢰하는
원본으로 계산하고 원본이 변경되면 재검토한다. 진단은 제작자에게 별도 채널로 전달받은
`bundle-manifest.json`의 해시를 요구한다. 패키지 전체와 내부 링크를 검사한 뒤, 검증한
바이트의 임시 복사본을 기존 진단기에 전달한다. 직렬화된 모델은 팀이 신뢰하는 실행에서 받는다.

### 사전 검사 명세

| 필드 | 입력 |
| --- | --- |
| `schema_version`, `kind`, `currency` | `1`, `synthetic` / `research` / `financial`, 평가 통화 |
| `window_id` | 변환 manifest의 Window ID. 기존 Team12 변환물에서 필수 |
| `training_config` | 실제 학습에 전달할 YAML/JSON 설명자. 사전 검사에서 상대 `data.dataset_dir`의 기준은 `ROOT` |
| `dataset` | `returns.parquet`, `rsi.parquet`, `macd.parquet`와 선택적인 `risk_tags.parquet`를 파일명별 설명자로 기록 |
| `dataset_manifest` | `team12-ppo-input-v1` 변환 manifest 설명자. 파일별 해시와 자산 순서가 입력과 일치해야 함 |
| `lineage`, `source_audit` | 피처 `lineage.json`과 원천 감사 결과 설명자. 간소화한 합성 fixture에서는 생략 가능 |
| `calendar`, `evaluation_policy` | 거래 일정과 평가 정책 기록 설명자 |
| `periods` | `[{ "name": "train", "start": "...", "end": "..." }, ...]`. Train, Validation, Test 순서의 시간대 포함 `[start,end)` 구간 |

거래 일정 JSON은 `schema_version: 1`, `source`, `status`, `start`, `end`,
`evaluation_at` 배열을 포함한다. 일정은 요청한 전체 구간의 예정 세션을 담는다.
`official`은 `evidence_ref`로 공식 근거를 기록하고, 미확인 일정은 `unverified`를 사용한다.
합성 일정은 `synthetic`을 사용한다. 실제 거래일을 판단할 때 해당 파일을 독립된 일정 자료에서
만든다. 가격 데이터의 날짜만 복사하면 누락을 검출할 근거가 사라진다.

평가 정책 JSON은 `schema_version: 1`, `version`, `status`, `calendar_sha256`를 포함한다.
상태는 `draft`, `unverified`, `approved` 중 하나이며 `approved`에는 `approval_ref`가
필요하다. 합성 정책은 `synthetic`을 사용한다. 검사는 제공된 기록의 필드와 해시를 확인한다.
공식 근거 문서나 승인 주체의 진위 확인은 팀의 별도 검토가 필요하다.

사전 검사는 일정과 데이터의 누락·추가 세션, 시간대, 자산 순서, 변환 당시의 구간 및 연속성
표식, RSI·MACD가 처음 관측되는 행, 에피소드 길이를 확인한다. Team12 인덱스는 서울 15:30 기준이다.
선택적 리스크 태그도 해시와 스키마를 검증한다. Train은 구간 안에서 관측 창을 채우며,
Validation과 Test는 직전 데이터로 관측 창을 채울 수 있다. 학습기의 포함형 `data.train_end`에는
Validation의 마지막 평가 시각을 넣는다. 학습 환경은 Validation 시작 전까지만 데이터를 사용한다.
진단 수익률은 수익률 종료 시각으로 구간을 구분하므로 첫 Validation 수익률의 경계가 모델 선택
콜백과 다를 수 있다. [PPO 진단 계약](ppo-diagnostics.md)을 참조한다.

| 상태 | 의미 |
| --- | --- |
| `ready` | 기술 검증을 통과한 합성 smoke 입력. `scope=synthetic_smoke` |
| `blocked` | 입력 불일치, 원천 감사 실패 또는 정식 학습 승인 보류 |
| `unavailable` | 명세 또는 필수 자료를 찾지 못함 |

`technical_status`와 `source_status`를 함께 확인한다. 기술 검사를 통과한 연구용 입력도
정식 학습은 보류한다. 현재 변환 계보는 학습 승인을 보류하는 계약이며, 이 도구의 결과에는
`formal_ready=false`와 `training_approved=false`가 남는다. `preflight`는 합성 smoke 준비 시
종료 코드 0, 보류 또는 자료 부재 시 2를 반환한다. 합성 smoke에 한해 간소화한 manifest를
지원한다. `schema_version: 1`, `dataset_kind: synthetic_test`, `training_approved: false`,
`asset_order`, 파일별 해시를 기록하고 계보를 제공하면 `source_lineage_sha256`도 기록한다.

### 완료 실행 전달 명세

사전 검사 명세에 다음 필드를 추가한다.

- `run_id`와 `training_manifest`. 원본 학습 manifest 파일의 상위 폴더가 `run_id`여야 한다.
- `checkpoint`. 설명자에 같은 `run_id`와 `kind`를 추가한다. `final`은
  `<run_id>/final_model.zip`, `best`는 학습 manifest의 Validation에 기록된
  `<run_id>/best_model.zip`을 지정한다.
- `monitor_files`. `<run_id>/monitor/env-000.monitor.csv`부터, 모든 병렬 환경의 로그 설명자.
- `curves`. `curves_csv`는 `<run_id>/curves.csv`, `convergence`는
  `<run_id>/convergence.json`, `plot`은 `curves/<run_id>.png`의 설명자.

`pack`은 학습을 실행한 환경의 원본 경로에서 수행한다. manifest에 기록된 경로로 체크포인트와
곡선이 동일한 실행 디렉터리에 속하는지 확인하고, Monitor와 manifest도 해당 실행 경로에 묶는다.
원본 설정과 manifest의 적용 설정, seed, 라이브러리 버전, 데이터 해시와 학습 범위를
대조한다. Monitor 헤더의 시작 시각과 곡선의 에피소드 기록도 대조한다. 곡선이 필요한 경우
원본을 수정하기 전에 실행 소유자와 생성 절차를 확인한다. 검증을 마치면 다음 파일을 새 폴더에 쓴다.

- `dataset/`, `evidence/`, `runs/<run_id>/`, `curves/`의 원본 복사본
- `training-config-effective.json`의 실제 적용 설정과 `handoff.json`의 이식 가능한 입력 명세
- `diagnostic.json`의 진단 입력 설명자
- `bundle-manifest.json`의 검증 결과와 파일별 해시, `SHA256SUMS.txt`의 검증 목록

원본 manifest와 설정 안의 과거 절대 경로는 기록으로 보존한다. 검증과 진단은 패키지 루트의
상대 경로로 파일을 찾는다. 동일한 원본을 다시 묶으면 생성 파일의 바이트가 같다. 기존 학습
manifest에는 체크포인트 해시가 없다. 전달 시 계산한 해시는 이번에 복사한 바이트의 증빙이며,
같은 원본 경로의 모델이 패키징 전에 교체됐는지는 별도의 학습 시점 해시나 보관 기록으로 확인해야
한다. 실행 소유자가 별도 경로로 전달한 manifest 해시는 `verify --manifest-sha256 HASH`로도
대조할 수 있다.
Linux의 `sha256sum -c SHA256SUMS.txt`로도 파일 바이트를 대조할 수 있다.

## 팀 통합과 남은 확인

기존 학습기, 큐, 재학습기, 데이터 변환기와 진단기 동작은 유지한다. PR #133 병합 후
통합 백테스트의 비교 명세에 패키지의 `diagnostic.json` 해시를 넣고, 공통 세션 CSV와
벤치마크 명세를 별도로 고정한다. 벤치마크, 비용, 날짜 경계를 포함한 합성 통합 테스트를
추가해 산출물 계약을 확인한다. PR #125의 평가 정책 승인, 실제 금융 원천의 검증과 학습 승인,
같은 run_id의 원본 모델·입력·설정·로그 전달이 정식 성과 검증의 선행 조건이다.
