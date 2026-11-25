# 실제 금융 뉴스 및 OpenDART 리서치 실행

네이버 뉴스는 Streamlit 검색 버튼을 누를 때 `GET /news`로 조회한다. 응답에는
`Cache-Control: no-store`를 설정한다. 뉴스는 리서치 문서 정제·벡터 색인·Gemini 입력·PPO
데이터에 전달되지 않는다. 화면에는 네이버 검색 표시와 게시 시각, 제공된 링크를 표시한다.

OpenDART 경로는 기존 수집·정제 계약을 사용한다. 공식 인증 공시검색 엔드포인트의 첫 페이지만
수집하고, 공시 날짜만 있는 경우 다음 날 0시 KST부터 모델이 사용할 수 있게 한다.
Gemini 계획·근거 검증·합성 뒤 반환하는 제목과 접수일은 인용된 공시 목록에서 직접 작성한다.
이 흐름의 `evidence_scope=dart_listing_metadata`는 본문 평가와 RiskTag 생성을 보류한다.

## 수집과 서비스 실행

Python 3.10에 프로젝트 의존성을 설치하고 저장소 루트에서 실행한다. 인증 파일은 기존
비공개 `.env`를 지정한다. Compose는 허용된 여섯 인증·모델 변수만 API 서비스에 전달한다.
실제 키가 들어 있는 Compose 전체 설정이나 컨테이너 환경 목록을 출력하지 않는다.

```powershell
$privateEnv = '<private env file>'
$endDate = (Get-Date).Date.AddDays(-1)
python scripts/prepare_dart_research.py --env-file $privateEnv --corp-code 00126380 `
    --start ($endDate.AddDays(-45).ToString('yyyyMMdd')) --end ($endDate.ToString('yyyyMMdd')) `
    --output artifacts/live-dart-listing
$env:API_RESEARCH_MANIFEST = '/app/artifacts/live-dart-listing/research_manifest.json'
$env:API_RESEARCH_MANIFEST_SHA256 = (Get-FileHash -Algorithm SHA256 `
    artifacts/live-dart-listing/research_manifest.json).Hash.ToLowerInvariant()
$env:API_PORT = '127.0.0.1:18161'
$env:DASHBOARD_PORT = '127.0.0.1:18561'
docker compose --env-file $privateEnv -p robo61local up --build -d --wait
```

합성 PPO 데모도 함께 표시하려면 기존 인수 테스트의 정책·체크포인트·백테스트·대시보드
산출물 환경변수를 먼저 설정한다. 실제 공시 리서치의 준비 상태는 해당 합성 PPO 모델과
독립적이다. [합성 인수 환경 실행 방법](end-to-end-integration.md)

Gemini는 `LLM_PROVIDER=gemini`와 `LLM_MODEL`의 설정 모델을 그대로 사용한다.
서비스는 작업을 직렬화하고 그래프 자동 재작성을 끈다. 완료된 일반 요청은 계획, 검증,
합성의 생성 호출 3번을 사용하며, 동일 질의·기준일·문서 버전·텍스트 모델의 성공 결과를
기존 캐시에서 재사용한다. HTTP 429, 인증, 모델 부재, timeout은 재시도 없이 표시할 수 있는
오류 코드로 남는다. [Gemini 한도 안내](https://aistudio.google.com/docs/rate-limits)

## GitHub Actions와 인증정보

기본 CI와 Docker 이미지 빌드는 모의 입력으로 검사하며 실제 API 비밀값을 사용하지
않는다. 실제 서비스 탐색에는 로컬의 제외된 `.env`를 사용한다. 팀이 향후 수동
실증 워크플로를 운영한다면 보호된 GitHub Actions 환경에
`NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`, `DART_API_KEY`, `LLM_API_KEY`를
비밀값으로 등록하고 실행 단계에만 환경변수로 전달한다. `LLM_PROVIDER`와
`LLM_MODEL`은 비밀이 아닌 변수로 관리할 수 있다. 실제 공시 ZIP, 가격, 체크포인트,
리서치 응답은 로컬에서 관리한다. Actions 로그와 이미지 빌드 인자로 인증정보를
전달하지 않는다.

본문 검증과 오프라인 연구는 별도 명령으로 재현할 수 있다.
[OpenDART 본문 근거 리서치](dart-body-research.md) · [공시 Event Study](dart-event-study.md)
