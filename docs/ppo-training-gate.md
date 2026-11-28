# PPO 연구 입력 게이트

`run_training()`은 CLI, 직접 호출, 재학습, 실험 호출 모두에서 같은 게이트를 통과해야 한다.
현재 `config/research_datasets.yaml`의 허용 목록은 비어 있으므로 금융 입력은 모두 차단된다.
허용 목록은 정식 학습 승인 목록이 아니다. 통과한 실행도 `training_approved=false`이고
성과는 연구 결과로만 다룬다.

## 연구용 데이터 등록

1. PR #134 형식의 preflight 명세를 만들고 `kind: research`, `dataset_id`를 기록한다.
   명세는 평가 일정, 정책, 변환 manifest, 원천 계보와 감사 결과, 학습 설정 및 각 파일의
   SHA-256을 고정한다. 학습 설정의 `data.dataset_dir`는 명세가 있는 디렉터리 기준 상대 경로다.
2. 팀이 입력 파일과 원천 감사, 미확인 사항을 검토한 뒤 저장소의
   `config/research_datasets.yaml`에 ID, 실제 PPO 입력 파일별 SHA-256, 허용 학습 시작·종료
   시각, 이 저장소의 검토 PR/Issue URL, 미확인 한계를 등록한다. 선택적인
   `risk_tags.parquet`가 있으면 그 해시도 등록한다. 변경은 코드 리뷰로만 병합한다.
3. 학습은 실제 적용 설정과 preflight 설정, 입력 바이트와 허용 목록 해시, 학습 구간을
   대조한 뒤 시작한다. `technical_status=passed`여도 원천 상태가 `failed`이면 차단한다.
   `unverified`는 한계를 명시한 검토 기록이 정확히 일치할 때만 연구용으로 허용한다.
   preflight의 최종 `status=blocked`는 정식 승인 보류를 뜻할 수 있으므로 기술 상태와
   원천 상태를 각각 확인한다.

```bash
python -m robo_advisor.rl.train --config ROOT/config.json \
  --dataset-dir ROOT/converted --preflight-spec ROOT/preflight.json
```

학습 요청자가 지정하는 명세 경로는 증거 위치일 뿐 허용 목록의 위치를 바꾸지 못한다.
검증 실패는 모델·환경·학습 산출물 디렉터리 생성 전에 중단된다. 학습 manifest에는
`input_gate`에 자료 유형, ID, 검토 근거, 입력 해시, 기술·원천 상태 및 한계를 기록한다.

## 합성 테스트와 신뢰 경계

기존 합성 smoke 테스트는 저장소의 지정된 pytest fixture 모듈이 만든 임시 입력의 해시를
명시적으로 등록한 경우만 허용한다. 등록과 학습 호출은 현재 실행 중인 동일한 테스트
모듈에서 이루어져야 한다. 등록 뒤 바이트가 바뀌면 재등록 전까지 차단한다.
PR #134의 핵심 입력 파일 구조 검사를 재사용하고 선택적 risk tag 파일도 해시를 고정한다.
학습 로더는 학습 구간에 포함되는 risk tag를 검증하며, 실제 로드한 파일 해시를 다시 대조한다.
완전한 합성 preflight 명세가 있는 fixture는 일정·구간 검사까지 수행한다.
운영 CLI는 연구용 preflight 명세를 요구하며 합성 명세를 거부한다. 따라서 CLI의
`--smoke`, 입력 manifest의 `training_approved` 선언, 임의의 JSON/YAML 경로만으로
합성 또는 연구용 권한을 얻을 수 없다. 합성 fixture 등록 경로는 저장소 테스트의 직접
함수 호출 전용이며 일반 합성 학습을 위한 운영 인터페이스가 아니다.

허용 목록과 fixture 모듈을 검토된 저장소 코드로 신뢰한다. GitHub PR/Issue URL은
형식과 저장소를 검사하지만 GitHub 리뷰의 실제 승인 여부를 실행 중 조회하지 않는다.
따라서 브랜치 보호, 필수 리뷰, 권한 관리가 신뢰 경계다. 로컬 저장소 설정이나 Python
프로세스 코드를 바꿀 권한이 있는 공격자를 이 소프트웨어만으로 막을 수는 없다.
정식 금융 학습 승인 발급·폐기 체계와 실제 금융 성과 검증은 이 게이트의 범위 밖이다.
