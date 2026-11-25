# 합성 PPO 통합 인수 테스트

`scripts/run_integration_smoke.py`는 기존 PPO 환경과 SB3 CPU 학습기로 합성 실행 하나를
생성한다. 가상의 시장 가격과 RiskTag는 업스트림 형식과 시점 규칙을 시험하는 데 사용한다.
기본 경로에는 외부 리서치 요청이나 금융 데이터 수집이 없다.

## 실행

Python 3.10과 저장소 의존성을 설치하고 프로젝트 루트에서 실행한다.

```powershell
python -m pip install --no-deps -e .
python scripts/run_integration_smoke.py --output artifacts/acceptance
```

실행은 입력을 사전 검사하고 PPO를 32스텝 학습한다. 패키지 manifest 해시를
`handoff-pin.json`에 따로 저장하고, 패키지를 새 경로로 옮긴 뒤 검증·진단한다.
백테스트는 패키지에서 검증된 파일만 임시 위치로 복사해 읽는다. 별도 합성 벤치마크
가격·거래일·환율에 기존 감사를 적용하고 PPO, MVO, 동일가중, 합성 시장 벤치마크를 비교한다.
비교 결과와 실행 manifest는 `comparison/`, 실제 평가 관측과 API 산출물은 `api/`에 생성한다.

다음 명령은 출력 경로 `artifacts/acceptance`를 사용한 예다. 포트와 프로젝트 이름은
로컬 환경에 맞게 지정한다.

```powershell
$acceptance = Get-Content -Raw artifacts/acceptance/api/acceptance.json | ConvertFrom-Json
foreach ($property in $acceptance.environment.PSObject.Properties) {
    $value = if ($property.Name.EndsWith('SHA256')) {
        $property.Value
    } else {
        "/app/artifacts/acceptance/api/$($property.Value)"
    }
    [Environment]::SetEnvironmentVariable($property.Name, $value, 'Process')
}
$env:API_PORT = '127.0.0.1:18161'
$env:DASHBOARD_PORT = '127.0.0.1:18561'
$env:API_RESEARCH_MANIFEST = ''
$env:API_RESEARCH_MANIFEST_SHA256 = ''
$env:LLM_API_KEY = ''
docker compose -p robo61local up --build -d --wait
python scripts/run_integration_smoke.py --output artifacts/acceptance --check-url http://127.0.0.1:18161
```

Streamlit은 `http://127.0.0.1:18561`로 열 수 있다. API 서비스는 파일을 읽기 전용으로
마운트하며 Streamlit은 HTTP로 산출물을 받아 표시한다. HTTP 인수 결과는
`http-verification.json`에 기록한다. 종료할 때는 동일 프로젝트에
`docker compose -p robo61local down`을 호출한다.

## 형식 및 검증

- 백테스트 CSV를 `strategy + window` 키의 `metrics.json`으로 변환한다. 동일 전략이
  중복되면 거부하고 불완전한 지표는 null로 제공한다. 패키지와 비교 manifest의
  체크포인트·실행·입력 해시가 모두 일치해야 한다.
- PPO 평가를 같은 옵션으로 재생하고 실제 환경 관측을 내보낸다. 재생 NAV와 비교 결과가
  일치해야 한다. 관측은 수익률 창, 기존 비중, RSI/MACD, 해당 시점에 사용 가능한
  RiskTag를 기존 관측 레이아웃으로 저장한다. HTTP 검사는 API 비중을 독립적인
  `PPO.predict`와 softmax 결과와 비교한다.
- SHAP은 실제 모델 학습 구간의 관측에서만 배경을 뽑고 마지막 평가 의사결정의 자산
  한 개를 설명한다. 포트폴리오·성과·리스크는 비교 NAV, 실제 비중, 학습곡선과 태그를 사용한다.
  NAV가 15% 하락 기준에 도달하거나 초과한 시점만 이벤트로 공개한다.
- ANOVA는 실험 설계를 충족할 다중 시드·윈도우 자료가 없어 unavailable로 표시한다.
  리서치가 미설정된 실서비스에서 `POST /research`와 상태 조회는 503을 확인한다.
  외부 LLM을 사용하는 성공 경로는 별도 인수 테스트에 해당한다.

이 작업의 입력·출력은 `synthetic`, `unverified`, `training_approved=false`다.
정식 평가에는 검증된 금융 데이터와 평가 일정, 승인된 평가 프로토콜, 해당 실행의 원본
체크포인트·학습 manifest·입력·Monitor, 통계 설계를 충족하는 다중 실행이 필요하다.

## 선행 변경과 평가 정책

통합 브랜치는 API와 Streamlit·컨테이너 기반(#126과 그 브랜치에 병합된
#127·#128), Compose 산출물 설정(#132), 비교 백테스트(#133), PPO 실행 재현
패키지(#134)에 의존한다. 선행 커밋은 해당 PR 이력으로, 통합 코드는 별도 커밋으로
확인할 수 있다. #125는 후보 평가 정책의 출처다. 이번 인수 테스트의
`training_approved=false` 상태는 실제 금융 자료와 평가 정책 검토가 끝날 때까지
유지한다.
