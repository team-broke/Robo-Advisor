# PPO 정책 설명 입력 계약 (#39)

`explain/policy.py`와 `explain/background.py`는 #40의 SHAP 계산에 넘길 함수와
학습 구간 배경집합을 만든다. SHAP 전체 계산·캐시(#40), 플롯(#41), API는 별도다.
실제 금융 데이터의 학습 승인이나 모델 성능 달성을 의미하지 않는다.

## 설명 대상과 출력 단위

현재 학습기의 일반 Gaussian PPO `MlpPolicy`를 대상으로 한다. 설명 함수는
`PPOPolicyWrapper.predict_mean_actions`이며, 확률 행동을 뽑지 않고 분포의
**원래 평균 행동값(raw mean logits)** 을 반환한다. 출력은 `(행 수, 자산 수+1)`이고
자산 열 순서 다음 마지막 열은 `CASH`다. `for_asset("SPY")`처럼 고른 함수는
`(행 수,)`를 반환하므로 KernelSHAP의 자산별 설명 대상으로 사용할 수 있다.

평균 행동값은 비중이나 퍼센트가 아니다. 현재 실행 경로와의 관계는 다음과 같다.

1. Gaussian 정책이 평균 행동값을 산출한다. 이것이 #39의 고정 설명 대상이다.
2. SB3 `predict(deterministic=True)`는 평균을 행동 공간 `[-20,20]`으로 자른다.
3. PortfolioEnv는 그 값을 softmax로 변환해 목표 비중을 만든다.

`predict_target_weights`는 2~3단계를 그대로 적용한 참조 함수다. 평균 행동값의
SHAP 기여도를 비중 변화량이라고 쓰거나, SHAP 값을 나중에 softmax해서 비중의
SHAP 값으로 해석하면 안 된다. softmax는 상대 차이에 반응하므로 raw mean의
증가 자체가 해당 자산 비중 증가를 뜻하지 않으며, clip 범위 밖 변화는 실행에
영향이 없을 수도 있다. 비중 자체를 설명하려면 #40에서 별도의 설명
대상·단위·캐시 키를 명시해야 한다.

예측은 eval/no-grad로 수행하고 종료 시 원래 정책의 training mode를 복구한다.
모델 가중치를 복사·동결하지 않으므로 설명 중 모델을 동시에 학습하거나 바꾸지
않는다. 기록할 모델 ID·체크포인트 해시는 후행 산출물에서 별도로 연결한다.
SDE·squashed 정책 및 연결된 VecNormalize는 지원하지 않는다. VecNormalize 정보가
없는 체크포인트에서 숨은 전처리 여부를 추론할 수 없으므로, 호출자는 현재
학습기의 정규화 래퍼 없는 입력 계약인지 확인해야 한다.

## 관측값의 순서와 변환

`ObservationLayout(assets, lookback)`는 PortfolioEnv의 실제 관측 순서를 명명한다.
전체 차원은 `A*(N+4)+1`이며, 예를 들어 자산 12개·lookback 20일이면 289차원이다.

| 순서 | 열 이름 | 단위·의미 |
| --- | --- | --- |
| 1 | `simple_return_lag_N-1__자산` … `simple_return_lag_0__자산` | 오래된 날짜부터 자산 순서로 펼친 단순수익률 |
| 2 | `weight__자산`, 마지막 `weight__CASH` | 현재 보유 비중, 목표 비중이 아님 |
| 3 | `rsi_fraction__자산` | 환경이 이미 RSI/100을 적용한 값 |
| 4 | `macd_price_normalized__자산` | 가격 정규화된 MACD 선, signal·histogram 아님 |
| 5 | `risk_score__자산` | 해당 시점에 사용 가능했던 위험 점수 |

`to_frame`과 `as_batch`는 행·열의 순서와 수치를 보존한다. scaler를 fit하거나
입력값을 재정규화하지 않는다. DataFrame은 정확한 열 순서, Series는 정확한 피처 인덱스를 요구하고, 배열은
1개 관측 또는 2차원 배치를 받는다. 복소수·문자·불리언·비유한 값과 다른 차원은
거부한다. 각 피처 그룹은 `feature_groups`에서 조회할 수 있다.

입력은 실제 환경의 `reset`/`step`에서 얻은 관측값이어야 한다. 72열
`features.parquet`를 그대로 넣거나, 모든 과거 행의 보유 비중을 현금 100%로
가정해서 대체하면 다른 설명이 된다. KernelSHAP의 가상 혼합 입력에 대응하기
위해 개별 행의 비중 합 같은 경제적 제약을 추가로 재투영하지 않는다. 피처 간
의존성·그룹 마스킹에 따른 해석 한계는 #40에서 다뤄야 한다.

## 재현 가능한 기준 관측 경로

실제 무작위 시작 학습 에피소드의 로그를 단순히 합치면 같은 날짜가 중복되거나
역순이 될 수 있다. 이를 날짜순으로 임의 중복 제거해 사용하지 않는다.
`collect_training_observations`는 **고정 체크포인트의 결정론적 기준 경로**를
실제 fit 구간에서 재생한다. 학습 에피소드 전체의 경험 분포와는 다르다.

- 자산 순서는 학습 manifest의 `data_cutoff.assets`, fit 경계는
  `data_cutoff.actual_first_index`와 `actual_last_index`를 사용한다. 마지막 값은
  검증 분리 후 실제 학습 마지막 행이다. 체크포인트와 manifest가 같은 실행에
  속하는지도 호출자가 확인한다. 모델의 공간 차원만으로 자산 순서는 알 수 없다.
- `evaluation_env_config`로 random start와 길이 제한을 끄고 seed=0으로 처음부터
  재생한다. 보유 비중은 실제 환경 전이로 갱신하며, 정책의 clip된 평균을 사용한다.
- MDD 보호 규칙은 유지한다. fit 끝 이전에 종료되면 부분 경로를 반환하지 않고
  오류로 중단한다. 안전장치를 몰래 끄거나 누락 날짜를 임의의 비중으로 채우지 않는다.
- 마지막 데이터 행은 다음 수익률이 없는 종료 상태이므로 설명할 결정 행에
  포함하지 않는다. 반환 Frame의 `attrs["collection"]`에 유효 환경 설정과
  정책 모드·fit 경계가 기록된다. 모델 체크포인트·해시도 별도로 함께 보존한다.
- 조기 종료 모델의 대체 기준 경로(여러 에피소드의 명시적 시작·보유 상태 선택 등)가
  필요하면 별도 계약을 정해야 한다. 현재 함수는 그 선택을 자동으로 하지 않는다.

## 학습 구간만 사용하는 층화 배경집합

`sample_training_background`는 관측 DataFrame과 같은 인덱스의 층 라벨 Series,
실제 모델-fit 시작·종료 시각(양쪽 포함), 표본 수, seed, `expected_strata`를 받는다.

- 인덱스 정렬·유일성·시간대 및 라벨 인덱스 일치 같은 **구조 검증은 전체 행에
  적용**한다. 구간 밖의 잘못된 인덱스도 오류이며, 이를 누수 불변성이라고 숨기지 않는다.
- 수치·라벨은 fit 구간을 자른 뒤 검사한다. 구조가 유효한 평가 구간의 값·라벨이나
  정상 인덱스의 추가 행은 결과에 영향을 주지 않는다. 제외 행의 문자열·결측 때문에
  전체 열이 object dtype이 된 경우에도 fit 내부의 실제 숫자만 추론해 검사한다.
- `expected_strata`에는 수집 대상 의사결정 날짜의 인과적 라벨에 근거한 기대 층을
  명시한다. 빠진 층이나 예상 밖 층이 있으면 실패한다. 누락된 하락 국면을 조용히
  제외한 채 충분한 배경집합이라고 반환하지 않는다.
- 각 층에서 적어도 하나를 뽑고, 작은 층은 전부 사용한다. 남은 수는 다른 층에
  균등하게 배분한다. 층 내부에서는 seed 기반 비복원 추출, 결과는 시간순이다.
  표본이 층 수보다 적거나 후보보다 많으면 실패한다.
- 시장 국면 등 라벨은 해당 시점까지의 정보만으로 만들어야 한다. 표집기가
  라벨 생성 과정의 미래 정보 사용 여부를 자동으로 증명할 수는 없다.
- 균등 층화는 실제 국면 출현 빈도와 다르므로 SHAP 기준값은 선택한 균형
  배경분포에 대한 값임을 보고한다.
- `BackgroundSample`은 선택 관측·라벨·seed·fit 경계와 `sample_id`를 제공한다.
  ID는 현재 값·열·시각·라벨·seed·경계의 SHA-256이며 모델 ID나 학습 승인 ID는 아니다.

**fit 경계는 호출자가 사실대로 제공해야 한다.** 상위 `data.train_end`가 검증
구간까지 포함한다면 사용하지 말고, 실제 fit 마지막 시각을 사용한다.

```python
from robo_advisor.explain.policy import ObservationLayout, PPOPolicyWrapper
from robo_advisor.explain.background import (
    collect_training_observations,
    sample_training_background,
)

# model: manifest와 같은 실행의 고정 체크포인트. dataset: 학습기의 TrainingDataset.
asset_order = manifest["data_cutoff"]["assets"]
actual_fit_start = manifest["data_cutoff"]["actual_first_index"]
actual_fit_end = manifest["data_cutoff"]["actual_last_index"]
layout = ObservationLayout(assets=asset_order, lookback=env_config.lookback)
wrapper = PPOPolicyWrapper(model, layout)
observations = collect_training_observations(
    model, dataset, env_config,
    layout=layout, train_start=actual_fit_start, train_end=actual_fit_end,
)
# causal_regime_labels는 미리 과거 정보만으로 계산한 Series다.
labels = causal_regime_labels.reindex(observations.index)
background = sample_training_background(
    observations, labels,
    layout=layout,
    train_start=actual_fit_start, train_end=actual_fit_end,
    expected_strata=expected_training_regimes,  # 실제 fit 의사결정 날짜에서 기대되는 층을 명시
    n_samples=30, seed=42,  # 후보 수·층 수에 맞춰 지정
)
model_fn = wrapper.for_asset("SPY")
x_background = layout.as_batch(background.observations)
y_mean = model_fn(x_background)
# #40에서 model_fn과 x_background를 KernelSHAP에 전달한다.
# 모델·코드·collection 설정·sample_id와 선택 행을 함께 보존한다.
```

## 검증 범위

합성 관측과 실제 SB3 PPO 정책으로 이름·수치 왕복, raw mean과 clip 경계,
실제 환경 목표 비중과의 일치, 저장·로드, 자산별 KernelSHAP 호출을 검사한다.
배경집합은 국면 불균형, seed 재현, fit 경계 밖 데이터 오염·추가에 대한 누수
회귀를 검사한다. 무작위 학습 설정에서도 기준 관측 경로의 시간순·유일성과
MDD 조기 종료 시 오류를 확인한다. 실제 금융 모델의 투자 성과나 SHAP 그림은 생성하지 않는다.

근거: [SB3 2.3.2 정책 predict의 clip 처리](https://stable-baselines3.readthedocs.io/en/v2.3.2/_modules/stable_baselines3/common/policies.html),
[Gaussian distribution의 deterministic mode](https://stable-baselines3.readthedocs.io/en/v2.3.2/_modules/stable_baselines3/common/distributions.html).
