# 요구사항 추적표

> 기준 브랜치/커밋: `develop` / `af170dc` (2026-11-25 확인)

상태 범례: ✅ 완료 · 🚧 진행 중 · ⏳ 후행 작업 대기 · ❌ 미구현

## 1. 문서 목적

이 문서는 프로젝트 명세의 필수 기능 요구사항과 리포트 섹션을 GitHub Issue, 실제 구현,
테스트, 증빙 산출물에 연결한다. 상태는 GitHub Issue의 열림/닫힘 여부가 아니라 위 기준 커밋에
포함된 코드·테스트·산출물과 열린 PR을 직접 확인한 결과로 판정한다.

- ✅ 완료: `develop`에 구현과 검증 근거가 모두 존재한다.
- 🚧 진행 중: 열린 PR에는 구현이 있으나 `develop`에는 아직 없다.
- ⏳ 후행 작업 대기: 일부 기반은 완료됐지만 명시된 후행 Issue의 구현·실험·산출물이 필요하다.
- ❌ 미구현: `develop`과 확인된 열린 PR에 구현 또는 필수 산출물이 없다.

## 현재까지 구현된 주요 기능

- 공통 도메인 스키마 / RiskTag 계약
- RSI / MACD
- 원천 데이터 검증
- 학습 구간 전용 정규화
- Gymnasium 포트폴리오 환경
- 거래비용 / 리밸런싱 회계
- 보상함수 3종
- MDD Safe-Guard
- Walk-Forward 분할
- 성과지표 12종
- Rolling Markowitz / 실패 대체 정책
- 시장 국면 라벨러
- 누수 검증 CI

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
| FR-01 | 자산·벤치마크·환율 데이터 수집과 원천 데이터 검증 | #9 · #10 · #11 | 🚧 진행 중 |
| FR-02 | 누수 방지 전처리, RSI·MACD, 학습 구간 정규화와 입력 계보 | #12 · #13 · #14 · #15 · #16 | 🚧 진행 중 |
| FR-03 | Gymnasium 관측·행동 공간, 거래비용·리밸런싱 회계, MDD Safe-Guard | #17 · #18 · #19 · #20 | ✅ 완료 |
| FR-04 | PPO 학습과 보상함수 3종·lambda 실험, 학습곡선 | #19 · #21 · #22 · #23 | 🚧 진행 중 |
| FR-05 | Walk-Forward 완전 재학습 백테스트와 12개 성과지표·비교 결과 | #24 · #25 · #26 · #27 | 🚧 진행 중 |
| FR-06 | 롤링 Markowitz, 실패 대체 정책과 MVO 백테스트 | #28 · #29 · #30 | 🚧 진행 중 |
| FR-07 | DART·뉴스 수집, 정제·시점 검증과 ChromaDB 색인 | #31 · #32 · #33 | 🚧 진행 중 |
| FR-08 | LangGraph Agentic RAG, self-correction, reasoning trace와 리스크 태그의 RL 연계 | #34 · #35 · #36 · #37 · #38 | 🚧 진행 중 |
| FR-09 | PPO 정책의 SHAP 설명과 Summary·Force Plot | #39 · #40 · #41 | 🚧 진행 중 |
| FR-10 | 시장 국면별 ANOVA 3종, 가정 진단과 Tukey HSD | #42 · #43 · #44 · #45 | 🚧 진행 중 |
| FR-11 | FastAPI 5개 엔드포인트, 사전 로드·캐시·오류 추적 | #46 · #47 · #48 · #49 | 🚧 진행 중 |
| FR-12 | HTTP 전용 Streamlit 대시보드와 Docker 배포 | #50 · #51 · #52 · #53 · #54 · #55 | 🚧 진행 중 |

### 2.2 상세

#### FR-01. 자산·벤치마크·환율 데이터 수집과 원천 데이터 검증

상태: 🚧 진행 중

관련 Issue: #9 · #10 · #11

구현: 시장 수집기와 불변 스냅샷을 구현했다.

테스트/검증: [`test_collectors.py`](../tests/test_collectors.py), [`test_snapshot.py`](../tests/test_snapshot.py), [`test_source_evidence.py`](../tests/test_source_evidence.py)

증빙 산출물: PR #107과 [원천 증빙 감사](source-evidence.md). Yahoo USD/KRW·KTB10Y 실패 및 원천 공개시각 미확인이 남아 있다.

#### FR-02. 누수 방지 전처리, RSI·MACD, 학습 구간 정규화와 입력 계보

상태: 🚧 진행 중

관련 Issue: #12 · #13 · #14 · #15 · #16

구현: RSI·MACD, 학습 구간 전용 정규화, 피처·계보 저장 및 PPO 입력 변환을 구현했다.

테스트/검증: [`test_indicators.py`](../tests/test_indicators.py), [`test_scaling.py`](../tests/test_scaling.py), [`test_feature_dataset.py`](../tests/test_feature_dataset.py), [`test_feature_audit.py`](../tests/test_feature_audit.py), [`test_ppo_input.py`](../tests/test_ppo_input.py)

증빙 산출물: PR #106·#114 및 [PPO 입력 준비 상태](ppo-data-readiness.md). 실제 입력의 원천 증빙이 필요하다.

#### FR-03. Gymnasium 관측·행동 공간, 거래비용·리밸런싱 회계, MDD Safe-Guard

상태: ✅ 완료

관련 Issue: #17 · #18 · #19 · #20

구현: [`environment.py`](../src/robo_advisor/rl/environment.py), [`accounting.py`](../src/robo_advisor/rl/accounting.py), [`safeguard.py`](../src/robo_advisor/rl/safeguard.py)

테스트/검증: [`test_environment.py`](../tests/test_environment.py), [`test_accounting.py`](../tests/test_accounting.py), [`test_safeguard.py`](../tests/test_safeguard.py)

증빙 산출물: 병합 PR #69 · #71 · #72

#### FR-04. PPO 학습과 보상함수 3종·lambda 실험, 학습곡선

상태: 🚧 진행 중

관련 Issue: #19 · #21 · #22 · #23

구현: PPO 학습기, 보상·λ 실험 매트릭스, 학습곡선·검증 체크포인트 선택을 구현했다.

테스트/검증: [`test_rewards.py`](../tests/test_rewards.py), [`test_training.py`](../tests/test_training.py), [`test_reward_sweep.py`](../tests/test_reward_sweep.py), [`test_curves.py`](../tests/test_curves.py), [`test_validation_selection.py`](../tests/test_validation_selection.py)

증빙 산출물: PR #91·#95·#96·#102·#130. 확정 입력의 보상별 실험표·곡선, 모델 원본이 필요하다.

#### FR-05. Walk-Forward 완전 재학습 백테스트와 12개 성과지표·비교 결과

상태: 🚧 진행 중

관련 Issue: #24 · #25 · #26 · #27

구현: Walk-Forward 윈도우별 완전 재학습·평가, 12개 성과지표와 기준 전략 원천 감사를 구현했다.

테스트/검증: [`test_walk_forward.py`](../tests/test_walk_forward.py), [`test_metrics.py`](../tests/test_metrics.py), [`test_retrain.py`](../tests/test_retrain.py), [`test_benchmark_audit.py`](../tests/test_benchmark_audit.py)

증빙 산출물: PR #119·#113 및 [벤치마크 감사](benchmark-data-validation.md). 공통 조건의 최종 비교표는 #27에서 생성한다.

#### FR-06. 롤링 Markowitz, 실패 대체 정책과 MVO 백테스트

상태: 🚧 진행 중

관련 Issue: #28 · #29 · #30

구현: 롤링 Markowitz와 실패 대체 정책, 월별 백테스트를 구현했다.

테스트/검증: [`test_mvo.py`](../tests/test_mvo.py), [`test_mvo_backtest.py`](../tests/test_mvo_backtest.py)

증빙 산출물: PR #75·#122. 공식 비교 기간의 성과표는 #27과 함께 생성한다.

#### FR-07. DART·뉴스 수집, 정제·시점 검증과 ChromaDB 색인

상태: 🚧 진행 중

관련 Issue: #31 · #32 · #33

구현: DART·RSS 수집, 문서 정제·중복 제거·시점 검사, ChromaDB 색인을 구현했다.

테스트/검증: [`test_research_sources.py`](../tests/test_research_sources.py), [`test_research_ingestion.py`](../tests/test_research_ingestion.py), [`test_research_vector_store.py`](../tests/test_research_vector_store.py)

증빙 산출물: PR #97·#98. 제출용 문서 및 색인 실행 증빙이 필요하다.

#### FR-08. LangGraph Agentic RAG, self-correction, 공개 reasoning trace와 리스크 태그의 RL 연계

상태: 🚧 진행 중

관련 Issue: #34 · #35 · #36 · #37 · #38

구현: LangGraph 근거 검증·재검색, 공개 실행 로그, RiskTag 변환과 관측 피처를 구현했다.

테스트/검증: [`test_research_graph.py`](../tests/test_research_graph.py), [`test_research_trace.py`](../tests/test_research_trace.py), [`test_research_risk_tags.py`](../tests/test_research_risk_tags.py), [`test_risk_features.py`](../tests/test_risk_features.py)

증빙 산출물: PR #117·#118·#124. 실제 자료의 리서치 요청에서 정책 결정까지 연결한 증빙이 필요하다.

#### FR-09. PPO 정책의 SHAP 설명과 Summary·Force Plot

상태: 🚧 진행 중

관련 Issue: #39 · #40 · #41

구현: 정책 래퍼, 학습 배경집합, SHAP 사전 계산·저장과 Summary·Force Plot 생성을 구현했다.

테스트/검증: [`test_policy_explanation.py`](../tests/test_policy_explanation.py), [`test_explanation_background.py`](../tests/test_explanation_background.py), [`test_shap_values.py`](../tests/test_shap_values.py)

증빙 산출물: PR #112·#116. 최종 모델의 SHAP 캐시와 플롯이 필요하다.

#### FR-10. 시장 국면별 ANOVA 3종, 가정 진단과 Tukey HSD

상태: 🚧 진행 중

관련 Issue: #42 · #43 · #44 · #45

구현: 시장 국면 라벨과 ANOVA 3종 표본, 가정 진단·조건부 Tukey HSD를 구현했다.

테스트/검증: [`test_regimes.py`](../tests/test_regimes.py), [`test_anova.py`](../tests/test_anova.py), [`test_anova_samples.py`](../tests/test_anova_samples.py)

증빙 산출물: PR #76·#121·#129. 확정 백테스트의 표본, 검정표와 그림이 필요하다.

#### FR-11. FastAPI 5개 엔드포인트, 사전 로드·캐시·오류 추적

상태: 🚧 진행 중

관련 Issue: #46 · #47 · #48 · #49

구현: PR #123으로 FastAPI 계약이 병합됐다. 사전 로드·캐시·추적은 열린 [PR #126](https://github.com/team-broke/Robo-Advisor/pull/126)에 있다.

테스트/검증: [`test_api.py`](../tests/test_api.py)

증빙 산출물: [PR #126 검사](https://github.com/team-broke/Robo-Advisor/pull/126/checks). 지연 검증은 #48, 실제 산출물 통합 검증은 #61에서 확인한다.

#### FR-12. HTTP 전용 Streamlit 대시보드와 Docker 배포

상태: 🚧 진행 중

관련 Issue: #50 · #51 · #52 · #53 · #54 · #55

구현: 6페이지 및 HTTP 클라이언트(#128), Compose·Docker 빌드(#127)가 열린 [PR #126](https://github.com/team-broke/Robo-Advisor/pull/126)에 있다.

테스트/검증: PR #126의 대시보드 API·페이지·HTTP 클라이언트 테스트 및 Docker CI

증빙 산출물: [PR #126 검사](https://github.com/team-broke/Robo-Advisor/pull/126/checks). 병합 후 실제 산출물로 전체 실행을 검증한다.

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
| RPT-06 | SHAP 해석 | #39 · #40 · #41 · #59 | ⏳ 후행 작업 대기 |
| RPT-07 | ANOVA 검증 | #42 · #43 · #44 · #45 · #59 | ⏳ 후행 작업 대기 |
| RPT-08 | 한계·개선 | #57 · #58 · #59 · #60 · #61 | ❌ 미구현 |

### 3.2 섹션별 근거

#### RPT-01. 개요·아키텍처

근거 코드/모듈: [`README.md`](../README.md)의 텍스트 아키텍처, 공통 계약 [`schemas`](../src/robo_advisor/schemas)

데이터·실험 산출물: 최종 아키텍처 그림·리포트 원고 예정

#### RPT-02. 데이터·전처리

근거 코드/모듈: [`validation.py`](../src/robo_advisor/data/validation.py), [`indicators.py`](../src/robo_advisor/features/indicators.py), [`scaling.py`](../src/robo_advisor/features/scaling.py), [CI 워크플로](../.github/workflows/ci.yml)

데이터·실험 산출물: 스냅샷·피처 계보와 연구 데이터 감사를 구현했다. 평가 입력의 공개시각·원천 검증이 남아 있다.

#### RPT-03. RL 환경 설계

근거 코드/모듈: [`environment.py`](../src/robo_advisor/rl/environment.py), [`accounting.py`](../src/robo_advisor/rl/accounting.py), [`safeguard.py`](../src/robo_advisor/rl/safeguard.py)

데이터·실험 산출물: 설계 도표·실행 결과 예정

#### RPT-04. 보상함수 3종 실험

근거 코드/모듈: [`rewards.py`](../src/robo_advisor/rl/rewards.py)

데이터·실험 산출물: 실험 실행기와 곡선 생성기를 구현했다. 확정 입력의 보상별 실험표·학습곡선이 필요하다.

#### RPT-05. 백테스트 결과

근거 코드/모듈: [`walk_forward.py`](../src/robo_advisor/backtest/walk_forward.py), [`metrics.py`](../src/robo_advisor/backtest/metrics.py), [`mvo.py`](../src/robo_advisor/baselines/mvo.py)

데이터·실험 산출물: PPO 재학습 및 MVO 백테스트 구현을 병합했다. #27에서 공통 조건의 최종 성과표를 생성한다.

#### RPT-06. SHAP 해석

근거 코드/모듈: [SHAP 플롯 생성기](../src/robo_advisor/explain/plots.py), PR #112·#116

데이터·실험 산출물: 최종 선정 정책의 Summary·Force Plot 필요

#### RPT-07. ANOVA 검증

근거 코드/모듈: 시장 국면 라벨러 [`regimes.py`](../src/robo_advisor/experiments/regimes.py)

데이터·실험 산출물: ANOVA 표본·가정 진단·조건부 Tukey 구현을 병합했다. 최종 실험의 표와 그림이 필요하다.

#### RPT-08. 한계·개선

근거 코드/모듈: [`README.md`](../README.md)에 제목만 있고 본문은 미작성

데이터·실험 산출물: 최종 분석·개선안 예정

## 4. 현재 미충족 항목

현재 `develop`에서 완료되지 않은 필수 항목과 해결 예정 Issue는 다음과 같다.

| 영역 | 남은 작업 | 관련 Issue |
|---|---|---|
| 데이터·전처리 | 수집·스냅샷·계보 원천 증빙과 실데이터 승인 | #9 · #11 · #15 |
| PPO·백테스트·MVO | 평가 규칙 승인, 윈도우·seed별 실험 및 공통 조건 성과표 | #1 · #27 |
| 리서치·SHAP·ANOVA | 실제 실험 입력과 실행 결과, 검정표·그림 확보 | #27 · #59 · #61 |
| API·대시보드 | PR #126 통합, 지연 측정, 실제 산출물로 서비스 연결 확인 | #47~#55 · #61 |
| 최종화 | 전체 검증, 코드 리뷰, README, 리포트·PDF, 재현성 검증 | #56 · #57 · #58 · #59 · #60 · #61 |

## 5. 유지보수 규칙

후속 PR이 `develop`에 병합될 때 이 문서도 함께 갱신한다.

1. 기준 커밋과 해당 요구사항의 상태를 업데이트한다.
2. `develop`에 실제로 추가된 구현 파일만 구현 모듈에 연결한다.
3. 실제 테스트와 생성된 데이터·모델·표·그래프·리포트 경로만 추가한다.
4. 열린 PR의 내용은 병합 전까지 진행 중으로 유지한다.
5. 코드만 있고 필수 실험·결과 산출물이 없으면 완료로 처리하지 않는다.
6. 근거 없는 경로·테스트명·산출물명을 예상해서 기록하지 않는다.
