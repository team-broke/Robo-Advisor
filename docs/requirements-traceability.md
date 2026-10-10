# 요구사항 추적표

> 기준 브랜치/커밋: `develop` / `804e575` (2026-10-08 확인)

상태 범례: ✅ 완료 · 🚧 진행 중 · ⏳ 후행 작업 대기 · ❌ 미구현

## 1. 문서 목적

이 문서는 프로젝트 명세의 필수 기능 요구사항과 리포트 섹션을 GitHub Issue, 실제 구현,
테스트, 증빙 산출물에 연결한다. 상태는 GitHub Issue의 열림/닫힘 여부가 아니라 위 기준 커밋에
포함된 코드·테스트·산출물과 열린 PR을 직접 확인한 결과로 판정한다.

- ✅ 완료: `develop`에 구현과 검증 근거가 모두 존재한다.
- 🚧 진행 중: 열린 PR에는 구현이 있으나 `develop`에는 아직 없다.
- ⏳ 후행 작업 대기: 일부 기반은 완료됐지만 명시된 후행 Issue의 구현·실험·산출물이 필요하다.
- ❌ 미구현: `develop`과 확인된 열린 PR에 구현 또는 필수 산출물이 없다.

## 2. 기능 요구사항 추적

### 공통 선행 계약

12개 기능 요구사항에서 함께 사용하는 도메인 계약은 #5에서
구현됐다. 포트폴리오 비중, UTC 시각, 출처, trace ID는
[`common.py`](../src/robo_advisor/schemas/common.py), 리스크 이벤트와 시점 계약은
[`risk.py`](../src/robo_advisor/schemas/risk.py)에 있으며,
[`test_common.py`](../tests/schemas/test_common.py)와
[`test_risk.py`](../tests/schemas/test_risk.py)가 검증한다. 병합된 PR #65가 현재 증빙이며,
별도 실행 산출물은 없다.

### 2.1 요약

| ID | 요구사항 | 관련 Issue | 상태 |
|---|---|---|---|
| FR-01 | 자산·벤치마크·환율 데이터 수집과 원천 데이터 검증 | #9 · #10 · #11 | ❌ 미구현 |
| FR-02 | 누수 방지 전처리, RSI·MACD, 학습 구간 정규화와 입력 계보 | #12 · #13 · #14 · #15 · #16 | 🚧 진행 중 |
| FR-03 | Gymnasium 관측·행동 공간, 거래비용·리밸런싱 회계, MDD Safe-Guard | #17 · #18 · #19 · #20 | ✅ 완료 |
| FR-04 | PPO 학습과 보상함수 3종·lambda 실험, 학습곡선 | #19 · #21 · #22 · #23 | ⏳ 후행 작업 대기 |
| FR-05 | Walk-Forward 완전 재학습 백테스트와 12개 성과지표·비교 결과 | #24 · #25 · #26 · #27 | ⏳ 후행 작업 대기 |
| FR-06 | 롤링 Markowitz, 실패 대체 정책과 MVO 백테스트 | #28 · #29 · #30 | 🚧 진행 중 |
| FR-07 | DART·뉴스 수집, 정제·시점 검증과 ChromaDB 색인 | #31 · #32 · #33 | ❌ 미구현 |
| FR-08 | LangGraph Agentic RAG, self-correction, reasoning trace와 리스크 태그의 RL 연계 | #34 · #35 · #36 · #37 · #38 | ⏳ 후행 작업 대기 |
| FR-09 | PPO 정책의 SHAP 설명과 Summary·Force Plot | #39 · #40 · #41 | ❌ 미구현 |
| FR-10 | 시장 국면별 ANOVA 3종, 가정 진단과 Tukey HSD | #42 · #43 · #44 · #45 | 🚧 진행 중 |
| FR-11 | FastAPI 5개 엔드포인트, 사전 로드·캐시·오류 추적 | #46 · #47 · #48 · #49 | ❌ 미구현 |
| FR-12 | HTTP 전용 Streamlit 대시보드와 Docker 배포 | #50 · #51 · #52 · #53 · #54 · #55 | ❌ 미구현 |

### 2.2 상세

#### FR-01. 자산·벤치마크·환율 데이터 수집과 원천 데이터 검증

상태: ❌ 미구현

관련 Issue: #9 · #10 · #11

구현: 미구현

테스트/검증: 미구현

증빙 산출물: 데이터 스냅샷 예정(#11)

#### FR-02. 누수 방지 전처리, RSI·MACD, 학습 구간 정규화와 입력 계보

상태: 🚧 진행 중

관련 Issue: #12 · #13 · #14 · #15 · #16

구현: RSI·MACD만 [`indicators.py`](../src/robo_advisor/features/indicators.py)에 존재한다. #14는 열린 PR #77에만 존재한다.

테스트/검증: [`test_indicators.py`](../tests/test_indicators.py). #14 검증은 `develop`에 없다.

증빙 산출물: RSI·MACD 병합 PR #68. 정규화·입력 데이터셋 산출물은 예정이다.

#### FR-03. Gymnasium 관측·행동 공간, 거래비용·리밸런싱 회계, MDD Safe-Guard

상태: ✅ 완료

관련 Issue: #17 · #18 · #19 · #20

구현: [`environment.py`](../src/robo_advisor/rl/environment.py), [`accounting.py`](../src/robo_advisor/rl/accounting.py), [`safeguard.py`](../src/robo_advisor/rl/safeguard.py)

테스트/검증: [`test_environment.py`](../tests/test_environment.py), [`test_accounting.py`](../tests/test_accounting.py), [`test_safeguard.py`](../tests/test_safeguard.py)

증빙 산출물: 병합 PR #69 · #71 · #72

#### FR-04. PPO 학습과 보상함수 3종·lambda 실험, 학습곡선

상태: ⏳ 후행 작업 대기

관련 Issue: #19 · #21 · #22 · #23

구현: 보상식만 [`rewards.py`](../src/robo_advisor/rl/rewards.py)에 존재하며 PPO 학습기는 미구현이다.

테스트/검증: [`test_rewards.py`](../tests/test_rewards.py). PPO·실험 검증은 미구현이다.

증빙 산출물: 보상식 병합 PR #70. 모델·실험표·학습곡선은 예정이다.

#### FR-05. Walk-Forward 완전 재학습 백테스트와 12개 성과지표·비교 결과

상태: ⏳ 후행 작업 대기

관련 Issue: #24 · #25 · #26 · #27

구현: 분할기 [`walk_forward.py`](../src/robo_advisor/backtest/walk_forward.py), 지표 [`metrics.py`](../src/robo_advisor/backtest/metrics.py). 재학습·결과 생성은 미구현이다.

테스트/검증: [`test_walk_forward.py`](../tests/test_walk_forward.py), [`test_metrics.py`](../tests/test_metrics.py)

증빙 산출물: 병합 PR #73 · #74. 백테스트 결과는 #27에서 생성할 예정이다.

#### FR-06. 롤링 Markowitz, 실패 대체 정책과 MVO 백테스트

상태: 🚧 진행 중

관련 Issue: #28 · #29 · #30

구현: `develop`에는 없다. #28·#29는 열린 PR #75에만 존재한다.

테스트/검증: `develop`에는 없음

증빙 산출물: MVO 백테스트 산출물 예정(#30)

#### FR-07. DART·뉴스 수집, 정제·시점 검증과 ChromaDB 색인

상태: ❌ 미구현

관련 Issue: #31 · #32 · #33

구현: 미구현

테스트/검증: 미구현

증빙 산출물: 문서 원문·색인 산출물 예정

#### FR-08. LangGraph Agentic RAG, self-correction, 공개 reasoning trace와 리스크 태그의 RL 연계

상태: ⏳ 후행 작업 대기

관련 Issue: #34 · #35 · #36 · #37 · #38

구현: RiskTag 계약과 환경의 risk score 입력 지점은 [`risk.py`](../src/robo_advisor/schemas/risk.py), [`environment.py`](../src/robo_advisor/rl/environment.py)에 존재한다. 생성·RAG·연계 데이터셋은 미구현이다.

테스트/검증: 스키마·환경 검증만 [`test_risk.py`](../tests/schemas/test_risk.py), [`test_environment.py`](../tests/test_environment.py)에 존재한다.

증빙 산출물: reasoning trace·리서치 결과·연계 데이터셋 예정

#### FR-09. PPO 정책의 SHAP 설명과 Summary·Force Plot

상태: ❌ 미구현

관련 Issue: #39 · #40 · #41

구현: 미구현

테스트/검증: 미구현

증빙 산출물: SHAP 캐시와 plot 예정

#### FR-10. 시장 국면별 ANOVA 3종, 가정 진단과 Tukey HSD

상태: 🚧 진행 중

관련 Issue: #42 · #43 · #44 · #45

구현: `develop`에는 없다. #43은 열린 PR #76에만 존재한다.

테스트/검증: `develop`에는 없음

증빙 산출물: ANOVA 표·진단·사후검정 시각화 예정

#### FR-11. FastAPI 5개 엔드포인트, 사전 로드·캐시·오류 추적

상태: ❌ 미구현

관련 Issue: #46 · #47 · #48 · #49

구현: 미구현

테스트/검증: 미구현

증빙 산출물: API 응답·지연 측정·로그 산출물 예정

#### FR-12. HTTP 전용 Streamlit 대시보드와 Docker 배포

상태: ❌ 미구현

관련 Issue: #50 · #51 · #52 · #53 · #54 · #55

구현: 패키지 골격 외 미구현

테스트/검증: 미구현

증빙 산출물: 대시보드 화면·Docker 빌드 결과 예정

## 3. 리포트 8개 섹션 추적

리포트 자체와 실험 산출물은 아직 생성되지 않았다. 아래의 기존 코드는 향후 원고의 근거이며,
산출물 경로는 실제 파일이 생기기 전까지 기록하지 않는다.

### 3.1 요약

| ID | 리포트 섹션 | 관련 Issue | 상태 |
|---|---|---|---|
| RPT-01 | 개요·아키텍처 | #58 · #59 · #60 | ⏳ 후행 작업 대기 |
| RPT-02 | 데이터·전처리 | #9 · #10 · #11 · #12 · #13 · #14 · #15 · #16 · #59 | 🚧 진행 중 |
| RPT-03 | RL 환경 설계 | #17 · #18 · #19 · #20 · #59 | ⏳ 후행 작업 대기 |
| RPT-04 | 보상함수 3종 실험 | #22 · #23 · #59 | ⏳ 후행 작업 대기 |
| RPT-05 | 백테스트 결과 | #25 · #27 · #30 · #59 | ⏳ 후행 작업 대기 |
| RPT-06 | SHAP 해석 | #39 · #40 · #41 · #59 | ❌ 미구현 |
| RPT-07 | ANOVA 검증 | #42 · #43 · #44 · #45 · #59 | 🚧 진행 중 |
| RPT-08 | 한계·개선 | #57 · #58 · #59 · #60 · #61 | ❌ 미구현 |

### 3.2 섹션별 근거

#### RPT-01. 개요·아키텍처

근거 코드/모듈: [`README.md`](../README.md)의 텍스트 아키텍처, 공통 계약 [`schemas`](../src/robo_advisor/schemas)

데이터·실험 산출물: 최종 아키텍처 그림·리포트 원고 예정

#### RPT-02. 데이터·전처리

근거 코드/모듈: [`indicators.py`](../src/robo_advisor/features/indicators.py)

데이터·실험 산출물: 원천 스냅샷·전처리 데이터·계보는 미구현

#### RPT-03. RL 환경 설계

근거 코드/모듈: [`environment.py`](../src/robo_advisor/rl/environment.py), [`accounting.py`](../src/robo_advisor/rl/accounting.py), [`safeguard.py`](../src/robo_advisor/rl/safeguard.py)

데이터·실험 산출물: 설계 도표·실행 결과 예정

#### RPT-04. 보상함수 3종 실험

근거 코드/모듈: [`rewards.py`](../src/robo_advisor/rl/rewards.py)

데이터·실험 산출물: 보상·lambda 실험표와 학습곡선은 미구현

#### RPT-05. 백테스트 결과

근거 코드/모듈: [`walk_forward.py`](../src/robo_advisor/backtest/walk_forward.py), [`metrics.py`](../src/robo_advisor/backtest/metrics.py)

데이터·실험 산출물: PPO·벤치마크·동일가중·MVO 비교 결과는 미구현

#### RPT-06. SHAP 해석

근거 코드/모듈: 미구현

데이터·실험 산출물: SHAP Summary·Force Plot 미구현

#### RPT-07. ANOVA 검증

근거 코드/모듈: `develop`에는 없다. 시장 국면 라벨러는 열린 PR #76에만 존재한다.

데이터·실험 산출물: ANOVA 표·가정 진단·Tukey HSD 시각화 미구현

#### RPT-08. 한계·개선

근거 코드/모듈: [`README.md`](../README.md)에 제목만 있고 본문은 미작성

데이터·실험 산출물: 최종 분석·개선안 예정

## 4. 현재 미충족 항목

현재 `develop`에서 완료되지 않은 필수 항목과 해결 예정 Issue는 다음과 같다.

| 영역 | 남은 작업 | 관련 Issue |
|---|---|---|
| 데이터 | 데이터 수집·검증·스냅샷 | #9 · #10 · #11 |
| 전처리 | 거래일 정합 전처리·정규화·입력 계보·누수 CI. #14는 PR #77에서 진행 중 | #12 · #14 · #15 · #16 |
| PPO | PPO 학습·보상/lambda 실험·학습곡선 | #21 · #22 · #23 |
| 백테스트 | 윈도우별 완전 재학습과 PPO·벤치마크·동일가중 결과 | #25 · #27 |
| MVO | MVO 구현·대체 정책·백테스트. #28·#29는 PR #75에서 진행 중 | #28 · #29 · #30 |
| 리서치 데이터 | DART·뉴스 수집부터 ChromaDB 색인 | #31 · #32 · #33 |
| Agentic RAG | LangGraph, self-correction, reasoning trace, RiskTag 생성·RL 연계 데이터셋 | #34 · #35 · #36 · #37 · #38 |
| SHAP | SHAP 설명·계산·시각화 | #39 · #40 · #41 |
| ANOVA | ANOVA 표본·국면·검정·사후 분석. #43은 PR #76에서 진행 중 | #42 · #43 · #44 · #45 |
| API | FastAPI·캐시·지연 테스트·로깅 | #46 · #47 · #48 · #49 |
| 대시보드·배포 | Streamlit 화면과 Docker 배포 | #50 · #51 · #52 · #53 · #54 · #55 |
| 최종화 | 전체 검증, 코드 리뷰, README, 리포트·PDF, 재현성 검증 | #56 · #57 · #58 · #59 · #60 · #61 |

## 5. 유지보수 규칙

후속 PR이 `develop`에 병합될 때 이 문서도 함께 갱신한다.

1. 기준 커밋과 해당 요구사항의 상태를 업데이트한다.
2. `develop`에 실제로 추가된 구현 파일만 구현 모듈에 연결한다.
3. 실제 테스트와 생성된 데이터·모델·표·그래프·리포트 경로만 추가한다.
4. 열린 PR의 내용은 병합 전까지 진행 중으로 유지한다.
5. 코드만 있고 필수 실험·결과 산출물이 없으면 완료로 처리하지 않는다.
6. 근거 없는 경로·테스트명·산출물명을 예상해서 기록하지 않는다.
