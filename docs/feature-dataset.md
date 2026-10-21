# Issue #15: 팀 12자산 모델 입력과 데이터 계보

이 모듈은 #12 정합을 다시 구현하지 않는다. 정합을 완료한 입력과 그 증거를 받아
#13 지표, #14 Train 전용 정규화, 다음 평가 구간 보상, #11 원천 계보를 연결한다.
PR #81의 미병합 코드를 복사하거나 현재 브랜치에 존재한다고 가정하지 않는다.

## 입력과 자산 순서

고정 순서는 `SPY, IWM, EFA, EEM, IEF, TIP, LQD, GLD, DBC, VNQ,
KODEX_200TR, KTB10Y`다. 기존 15자산 실험 설정과 기준100 가격지수는 이 입력을
대신하지 않는다. 입력은 실제 원화 평가가격, 원천 스냅샷 manifest SHA-256,
자산별 원천 검증 상태, 정합·환율 정책, 실제 공개시각 증거를 함께 전달해야 한다.

현재 연구자료는 KTB10Y가 `failed`, 나머지 11개가 `unverified`다. Yahoo `KRW=X`와
역사적 `available_at`도 없다. 따라서 현재 실제 자료로 학습 승인된 모델 입력을
만들 수 없다. 합성 데이터의 기능 검증 결과는 실제 금융 자료 검증과 구분한다.

`alignment` 인수는 #12 `AlignmentResult`와 같은 속성을 제공하는 독립 protocol이다.
`prices_krw`, `log_returns`, `source_dates`, `available_at`, `forward_filled`는 자산별
DataFrame이다. `fx`는 `rate`, `observation_date`, `available_at`, `forward_filled` 열을
제공하고, `evaluation_at`은 UTC aware Series, `return_start_dates`는 각 수익률의
실제 시작 관측일 Series다. `diagnostics`도 함께 전달한다. 인덱스와 자산 집합이
서로 일치해야 하며 미래 공개시각 또는 호환되지 않는 정합 정책은 거부한다.

정합 증거 `alignment_evidence`는 원천 manifest SHA-256, 검증 상태, 공식 캘린더
상태, 실제 공개시각 상태, 환율 원천, 가격 정의를 전달한다. 현 계약의 캘린더는
`KRX_US_intersection`, 해외 종가 래그는 교집합을 만들기 **전의 전체 미국 거래일**
기준 1세션, Forward Fill 한도는 5세션, 환율 최대 관측 연령은 7일이다. 이 모듈은
이 정책을 직접 실행하거나 누락 증거를 만들어 넣지 않는다. 실제 자료에는
`yahoo:KRW=X` 출처와 검증된 `USD_KRW` 원천 스냅샷도 필요하다.

`dataset_kind="synthetic_test"`는 #11 원천 12종 모두 `source="synthetic"`인 실제
테스트 스냅샷일 때만 허용한다. 실제 연구자료의 상태를 합성으로 바꿔 우회할 수 없다.
`research_only` 입력도 현재의 failed/unverified 원천으로는 차단된다. 모든 산출물은
`training_approved=false`를 유지한다.

## 인과적 피처와 워밍업

자산별 로그수익률, RSI(14), 가격 정규화 MACD(12,26), signal(9), histogram,
20관측 로그수익률 변동성을 생성한다. 자산 우선·피처 순서의 72개 열을 유지한다.
MACD 구현이 초기부터 값을 반환하더라도 #15는 초기 33관측을 명시적인 워밍업으로
다룬다. 워밍업 또는 결측 때문에 사용할 수 없는 행은 기록하고 학습 입력에서
제외한다. 미래값 참조, 결측 보간, Forward Fill, 환율 변환, 미국 종가 래그는 여기서
수행하지 않는다. 실제 공개시각을 수집시각이나 관측일로 만들어 넣지 않는다.

`return_start_dates`가 직전 생존 관측일과 다른 지점은 새 연속 구간으로 나눈다.
그 구간에서도 첫 33관측을 새 워밍업으로 마스킹하여 사라진 중간 관측을 지표
계산이 가로지르지 않는다. `cutoff_at`을 주면 UTC aware 평가시각 기준으로 먼저
잘라낸 뒤 피처·보상·정규화를 계산한다. 잘려 나간 미래 행으로 마지막 보상을
확정하지 않는다. 열 이름은 `SPY__log_return`과 같이 `자산__피처` 형태다.

## 분할과 정규화

Train, Validation, Test는 명시된 반개구간 경계를 사용한다. 워밍업 문맥은 과거
지표 계산에만 사용하며 Train 통계에 자동 포함하지 않는다. 각 Walk-Forward
윈도우는 새 `TrainOnlyStandardScaler`를 만들고 해당 Train만으로 fit한다.
Validation/Test는 transform만 한다. 자산·피처 열 순서와 Train 평균·scale을 별도로
기록하여 모델 입력을 재현할 수 있게 한다.

## 보상과 시장 국면

다음 평가 구간 수익률은 관측 피처 파일에 포함하지 않는다. 평가구간 종료시점이
확정되지 않은 마지막 행의 보상은 NaN이며, 분할 경계를 넘어가는 보상은 해당
분할에서 사용하지 않는다. 보상을 생성할 수 있는 것과 거래비용·현금·Rf를 포함한
PPO 환경의 최종 보상 정책은 별도 계약이다.

`rewards.parquet`는 다음 평가 구간의 **자산별 로그수익률**을 보존한다. 현재 관측
피처에 다음 수익률을 섞지 않는다. Window의 `splits.parquet`에서 현재와 다음 행이
같은 분할에 속할 때만 `reward_eligible=true`다. 마지막 행 또는 불연속 구간의
보상은 NaN이다. PPO가 단순 포트폴리오 수익률을 필요로 하면 별도 환경에서
로그수익률 의미를 확인하고 변환해야 하며 거래비용·현금 규칙은 여기서 추정하지 않는다.
각 Window의 별도 `rewards.parquet`에는 이 사용 가능 마스크를 적용하여 경계를
넘는 보상을 NaN으로 저장한다. 루트 보상은 계보 분석용 전체 다음 수익률이며
모델 학습 시 Window별 보상과 분할 마스크를 함께 사용한다.

시장 국면은 검증된 벤치마크, 실제 거래 캘린더와 명시적인 block 정책을 제공할 때만
기존 `label_market_regimes`와 연결한다. 필요한 증거가 없으면 라벨을 만들지 않고
누락 이유를 기록한다. ETF 수익률을 임의로 벤치마크로 대체하지 않는다.

`BenchmarkInput`의 evidence는 별도 검증기가 확인한 원천·공개시각·정합 증거를
호출자가 전달하는 계약이다. `status="passed"`라는 문자열을 넣는 것만으로 외부
원천을 독립 검증해 주는 기능은 아니다. 현재 실제 벤치마크의 역사적 공개시각
증거가 없으므로 실제 국면 라벨은 생성하지 않는다.

## 저장과 검증

`features.parquet`와 별도 보상·분할·정규화 산출물을 기록하고 `lineage.json`에 원천
manifest SHA-256, 피처 버전, 자산·열 순서, 기간, 컷오프, 워밍업, 정합·환율 정책,
검증 상태를 남긴다. 저장 성공은 원천 오류 해소 또는 학습 승인을 의미하지 않는다.
원본 스냅샷·연구자료를 수정하지 않는다. 저장된 파일의 SHA-256과 논리 fingerprint를
읽을 때 검증하고, 기존 출력 ID를 덮어쓰지 않는다.

```text
{output_root}/{dataset_id}/
  features.parquet             # 원척도 피처 72열, 워밍업은 NaN
  rewards.parquet              # 다음 평가 구간 로그수익률 12열
  rows.parquet                 # 워밍업·연속 구간·공개시각·보상 확정 여부
  windows/{window_id}/
    features.parquet           # 해당 Train scaler로 transform한 모든 분할
    splits.parquet             # train/validation/test와 관측·보상 사용 가능 여부
    rewards.parquet            # 분할 경계·워밍업 보상을 NaN으로 마스킹
  lineage.json                 # 원천, 정책, 스케일 통계, 파일 해시, 완료 표시
```

`regimes.parquet`는 검증된 벤치마크 입력이 있을 때만 추가된다. Window별 열 순서,
평균, scale, fit 행 수와 기간은 lineage의 `windows`에 기록한다. `lineage.json`이
최종 완료 표시이므로 중간 실패한 폴더는 완성된 데이터셋으로 읽히지 않는다.
외부에서 보관한 `expected_lineage_sha256`을 읽기에 전달하면 lineage 자체의
변경도 탐지한다. 내부 해시만으로 공격자가 모든 파일과 lineage를 함께 다시 쓰는
상황까지 인증할 수는 없다.

합성 테스트 산출물의 lineage 일부는 다음과 같다. 이 표시는 실제 금융 데이터의
검증 결과가 아니다. 실제 파일에는 원천 SHA-256, 기간, cutoff, 12자산 순서,
72개 피처 열, Window별 scaler와 저장 파일 descriptor가 함께 들어간다.

```json
{
  "schema_version": 1,
  "feature_version": "team12-causal-v1",
  "dataset_kind": "synthetic_test",
  "training_approved": false,
  "validation": {"status": "synthetic_only"},
  "reward_kind": "next_evaluation_log_return",
  "market_regimes": {
    "status": "not_generated",
    "reason": "verified_benchmark_not_supplied"
  }
}
```

## API 연결 예시

```python
from robo_advisor.data.snapshot import read_snapshot
from robo_advisor.features.dataset import (
    DatasetWindow, assess_dataset_readiness, build_feature_dataset,
)
from robo_advisor.features.dataset_io import write_feature_dataset, read_feature_dataset

snapshot = read_snapshot(snapshot_path, expected_manifest_sha256=trusted_manifest_sha256)
readiness = assess_dataset_readiness(
    snapshot,
    expected_manifest_sha256=trusted_manifest_sha256,
    alignment_evidence=evidence,
)
# 현재 실제 12자산 스냅샷의 readiness는 blocked다. reasons를 먼저 확인한다.

# 향후 검증된 #12 adapter를 전달할 때의 호출 형태:
result = build_feature_dataset(
    alignment,
    source_snapshot=snapshot,
    expected_manifest_sha256=trusted_manifest_sha256,
    windows=[DatasetWindow("wf-1", train_start, train_end, validation_end, test_end)],
    alignment_evidence=evidence,
    dataset_kind="research_only",
    cutoff_at=cutoff_utc,
)
output = write_feature_dataset(result, output_root, dataset_id="reviewed-team12-v1")
loaded = read_feature_dataset(output, expected_lineage_sha256=trusted_lineage_sha256)
model_inputs = loaded["frames"]["windows/wf-1/features"]
eligible = loaded["frames"]["windows/wf-1/splits"]["eligible_observation"]
```

예시의 시각·경계·정합·증거 변수는 호출자가 제공하는 실제 값이며 자동 생성하지
않는다. `DatasetBlockedError`에는 차단 사유가 포함된다. 예외를 무시하고 상태를
`passed`로 덮어쓰지 않는다. 저장 API는 `FeatureDataset(frames, lineage)`와 같은
결과 객체를 받아 모든 DataFrame을 native Parquet으로 쓰고 왕복 fingerprint를
검증한다. 임시 설치한 Parquet 엔진의 경로는 프로젝트 코드에 넣지 않는다.

## 오프라인 검증 범위

`tests/test_feature_dataset.py`는 테스트 중 모든 외부 socket 연결을 차단한다.
합성 12자산 #11 스냅샷으로 실제 Parquet 왕복, SHA-256 손상, 추가·누락 파일,
덮어쓰기 차단, 부분 저장 롤백을 확인한다. 미래 가격·Validation/Test 변동이 과거
피처와 Train 통계에 영향을 주지 않는지, split 경계 보상 제거, 실제 공개시각 위반,
failed/unverified 원천 차단과 합성 자료로의 위장 차단을 확인한다. 이 검증은
실제 금융 데이터의 공개시각 또는 가격 조정 정의를 검증한 것으로 해석하지 않는다.

2026-10-16의 별도 작업 폴더 검증에서 신규 테스트 **55개**, 그중 누수 마커 테스트
**7개**가 통과했다. 테스트 파일의 Black·flake8 검사도 통과했다. 전체 프로젝트
회귀 검증 수와 실제 자료 readiness 결과는 별도 실행 보고서에서 확인한다.

## 남은 실제 자료 검증

1. KTB10Y OHLC 오류를 정정본 원천과 비교하고 재검증한다. 원본 오류를 자동 수정하지 않는다.
2. 국내 ETF 관측일 차이를 공식 시장 캘린더와 대조한다.
3. Yahoo `KRW=X` 확보 및 KRW per USD 방향·공개시각 검증이 필요하다. FRED DEXKOUS를 같은 원천으로 간주하지 않는다.
4. 각 원천의 실제 역사적 공개시각, 배당·분할 조정 정의를 독립 증거로 검증한다.
5. #12 정합 구현이 병합되면 이 독립 인터페이스를 명시적으로 연결하고 동등성 테스트를 수행한다.

