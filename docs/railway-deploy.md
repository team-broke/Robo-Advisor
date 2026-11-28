# Railway 배포

같은 저장소로 Railway 서비스 2개(`api`, `dashboard`)를 만든다. 로컬 `docker compose`와
같은 Dockerfile을 쓰며, 두 이미지는 플랫폼이 주입하는 `PORT`를 따르고 없으면 8000/8501을 쓴다.

## 서비스 구성

| 항목 | `api` | `dashboard` |
|---|---|---|
| Config file path (Settings → Config-as-code) | `/deploy/railway/api.json` | `/deploy/railway/dashboard.json` |
| Dockerfile | `Dockerfile.api` | `Dockerfile.dashboard` |
| Healthcheck | `/health` | `/_stcore/health` |
| 공개 도메인 | 만들지 않음 (내부망만) | Generate Domain |
| 필수 변수 | `PORT=8000` | `PORT=8501`, `API_BASE_URL=http://api.railway.internal:8000` |

- 서비스 이름을 `api`로 둬야 `api.railway.internal` 주소가 맞는다. 이름을 바꾸면
  `API_BASE_URL`도 같이 바꾼다.
- `PORT`를 직접 지정하면 내부망 포트가 고정돼 대시보드의 `API_BASE_URL`과 맞출 수 있다.
- 배포 브랜치는 각 서비스 Settings → Source에서 `develop` 또는 `main`으로 지정한다.
- `watchPatterns` 때문에 해당 서비스와 관련된 파일이 바뀐 커밋에서만 다시 배포된다.

## 사전 계산 산출물

compose의 `./artifacts`, `./models`, `./data` 바인드 마운트는 Railway에서 쓸 수 없고,
`.dockerignore`에서 제외돼 이미지에도 들어가지 않는다. 산출물이 없으면 서비스는 정상 기동하지만
`/health`에 미로드 사유가 표시되고 해당 엔드포인트는 503을 반환한다.

산출물을 붙이려면 `api` 서비스에 Volume을 하나 만들고(서비스당 1개) 한 경로 아래에 모은다.

1. `api` 서비스 → Volume 추가, 마운트 경로 `/app/store`.
2. `/app/store/artifacts`, `/app/store/models` 등에 최종 산출물을 올린다.
3. `docker-compose.yml`의 `API_*` 변수를 `/app/store/...` 기준 경로와 신뢰 SHA-256으로 지정한다.

컨테이너는 uid 10001로 실행된다. Volume은 root 소유로 마운트되므로 올린 파일을 읽지 못하면
`RAILWAY_RUN_UID=0`을 지정하거나 파일 권한을 읽기 가능하게 맞춘다. API는 산출물을 읽기만 한다.

## 주의

- 대시보드에는 인증이 없다. 공개 도메인을 열면 누구나 화면과 대시보드를 통한 API 호출(`/research` 등)을 쓸 수 있다.
- `LLM_API_KEY`를 넣으면 외부에서 `/research`를 호출할 때마다 비용이 발생할 수 있다.
  데모 용도에서는 비워 두거나 접근을 제한한다.
- `API_*` 경로나 해시가 바뀌면 `api` 서비스를 재배포해야 반영된다.
