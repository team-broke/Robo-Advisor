# #9/#11 원천 증빙 품질 보고서

2026-10-28 KST 로컬 보존 자료를 읽기 전용으로 재확인했다. 아래 초기 수치는
CLI 구현 전 직접 inspection한 결과다. 신규 CLI의 실행 결과는 최종 실행 후
별도 기록한다. 가격·snapshot·학습 승인 상태를 변경하지 않았다.

`<evidence-root>`는 보존한 `team12-experimental-<delivery-id>` 패키지다.
`config/team12_source_evidence.json`은 이 루트 안의 상대 경로와 금융 증빙의
SHA-256만 지정한다. `base_audit_config`는 저장소 `config/team12_audit.json`을
재사용한다. 줄바꿈에 따라 달라지는 Git 설정 파일 자체의 해시를 금융 원천
인증값으로 삼지 않는다. 원문·개인 경로·대용량 산출물은 Git에 추가하지 않는다.

## KTB10Y 원본과 기존 별도 정정본

부모 snapshot은 `failed`, 기존 연구 선택본과 KRX 후보는 `unverified`다.
2020-03-10 부모는 Open=127475, High=127500, Low=127500, Close=127500,
Volume=28091로 Open < Low다. 모순은 정확히 한 행이며 data 위치 127,
raw 위치 291, 헤더를 포함한 원 CSV 293행이다. 선택본·KRX 후보의 Low는
126795이고 나머지 OHLCV는 같다. 이번 작업에서 새 정정을 적용하지 않았다.

부모 원 CSV의 실제 SHA는
`79e5bdc53e1d4f3d3b90aa6f8ddaac6e405f1b6d2c5942d9b01b137a612351dd`다.
1,692행의 문자열 셀 전체가 부모 raw_data와 같다. 부모 선택 data는 1,528행,
기존 선택본 raw_data도 1,528행으로 부모 data와 정확히 같다. 선택본의
`parent_validation`은 부모 검증과 같고, 달라진 data 셀은 Low 하나뿐이다.
요청 범위 밖 164행의 보존 근거는 부모 raw_data와 원본 CSV다.

설정의 `ktb_original`은 Yahoo CSV가 아니라
`evidence/official-KRX-originals/KRX_148070_20200310.csv`라는 KRX 비교 원문이다.
353바이트, SHA
`95a6bced1f66c5429524ee23f56354742e8ff4194ca9db739997d457e137074e`이며
CP949로 읽은 OHLCV가 후보 snapshot과 같다. **이 날짜의 receipt JSON은 없다.**
기존 `official_values.evidence_status=verified` 선언을 수집 receipt 확인으로
해석하지 않는다. `collected_at=None`, `EXACT_COLLECTION_TIME_NOT_RECORDED`와
수집시각 미확인 사유를 유지한다. 같은 폴더의 다른 날짜 request JSON 52개는
지정 CSV SHA·바이트 수와 52/52 일치하지만 이 정정 날짜의 receipt가 아니다.

## Yahoo USD/KRW 원문 대조

부모 `USD_KRW`의 data는 1,653행, raw 및 원 JSON timestamp는 1,654행이다.
`fx/USD_KRW/raw_text.txt`의 실제 SHA는
`9fadca9aaa6dfb43bb1395f1e30f0fa0fb3db48318e3f404b485d760121124ef`다.
요청 범위 `[2019-09-01, 2026-01-01)`의 OHLCV 전체가 원 JSON quote와 값·NaN까지
정확히 같다. 따라서 아래 결함은 정제 단계에서 새로 생성된 값이 아니다.

| 집계 단위 | 실측 |
|---|---:|
| OHLC 모순 고유 날짜 | 156 |
| 범위 내 OHLC/Volume 전체 null 날짜 | 4 |
| 가격 결측 셀 | 16 = 4일 × OHLC 4열 |
| Volume 결측 셀 | 4 |
| 범위 밖 raw null 날짜 | 1 = 2026-01-01 |

범위 내 null 날짜는 2025-01-01, 04-18, 04-21, 12-25이며 위치는 각각
1392, 1469, 1470, 1648이다. 20개 결측 셀을 20개 결측일로 세지 않는다.
모순 156일의 처음/끝은 2019-09-12/2025-12-29다. 2020-03-10에는 원문과 data
모두 Open=1198, High=1199.68994140625, Low=1189.1400146484375,
Close=1199.9599609375여서 Close > High다. 구체적인 공급자 오류 원인은 미확인이다.

원 JSON의 currency=KRW, symbol=KRW=X, longName=USD/KRW는 단위의 근거다.
`exchangeTimezoneName=Europe/London`으로 timestamp를 일자화하면 raw의
ProviderBarDate와 정확히 같다. metadata의 Asia/Seoul 수집 계약과 구별한다.
현재 수집시각을 과거 available_at으로 해석하지 않는다.

## 독립 FX 증빙과 남은 차단

SMBS MAR·ALFRED DEXKOUS는 Yahoo OHLC를 고친 값이 아니라 별도 fixing과
시나리오다. 다음은 별도로 수행한 FX 원문 감사의 실측 범위이며,
아직 신규 CLI의 최종 통합 실행 완료를 뜻하지 않는다.

- SMBS 1,561행/결측 0: 선택한 연간 표 7개의 원문과 정확히 일치. 색인에 있는
  선택 MAR와 무관한 원응답 3개는 누락돼 partial/unverified로 구별한다.
- ALFRED 1,599행/결측 20: 선택한 331개 빈티지 응답과 정확히 일치.
  색인 729파일·358 receipt 중 HTTP 200은 357개, H10 about의 403 기록은 1개다.
  실패 응답을 성공 증빙으로 사용하지 않는다.

색인 SHA는 SMBS
`00562651fa961e9263e049c7c1402326dcdef214c9ef8fbec5fe3f6f5f437194`, ALFRED
`a426c30a2f788d31f272a7ed774dd5e82c83e935e4335a794ed575d28ccdc518`다.
해시 일치는 보존 자료의 동일성을 뜻하며 공식 전송 인증이나 과거 공개를 증명하지
않는다. 실제 공개시각·배당/분할/수정주가 빈티지·전 기간 공식 일정은 미확인이다.
부모 failed 상태와 모든 `training_approved=false`를 유지한다.

#94 변환기·원척도 후행 수익률/미래 target 분리·시간대·결측 에피소드·체결 정책이
남아 있으므로 승인된 PPO 학습 준비 완료를 주장하지 않는다. 관련 계약은
[기존 감사](data-audit.md), [실측 인벤토리](real-data-inventory.md),
[#94 연결 준비](ppo-data-readiness.md)에 있다.

## 신규 CLI 최종 실행과 회귀 검증

2026-10-28 KST, develop 기준 `6d80cd22dbc6efc43bf88c2f475ac02f46aacdf3`에서
분리한 `feat/9-11-source-evidence`의 Python 3.10.21 환경에서 확인했다.
최종 로컬 보고서는 ignored 경로
`artifacts/data-validation/source-evidence-20261028-handoff.json`에 보존했다.
버전은 `team12-source-evidence-v1`이며 부모·선택·KRX 후보 manifest SHA와
SMBS/ALFRED CSV·증빙 index SHA를 함께 기록한다. 이 해시의 일치는 공급자
원문의 역사적 진실성이나 행별 공개시각 인증을 뜻하지 않는다.

| 검사 | 실제 결과 |
|---|---|
| 기존 데이터 감사 재사용 | 부모 결함에 따라 failed 유지 |
| KTB 부모/선택/후보의 값 대응 | Low 한 셀만 변경, 부모 raw/validation 보존 검사 passed |
| KTB 단일 CSV 후보 | OHLCV 일치, receipt 미제공으로 unverified |
| Yahoo JSON/raw/범위 선택 | 값·날짜 대응 passed, OHLC 156일 및 결측 4일로 failed 유지 |
| SMBS 원문 대조 | 1,561행/7개 응답 일치, HTTP 및 optional 응답 3개 부재로 unverified |
| ALFRED 원문 대조 | 1,599행(20 null)/331개 빈티지 응답 일치, 실제 공개는 unverified |
| 종합·학습 상태 | failed, training_approved=false, PPO blocked |
| 신규 pytest | 48 passed: 가격 16 / FX 증빙 16 / 원문·CLI 16 |
| 전체 pytest | 1,239 passed, 0 failed, 6 warnings |
| leakage 선택 실행 | 64 passed, 1,175 deselected |
| Black / flake8 | 전체 통과, Python 파일 92개 |

경고 6개는 기존 Gymnasium checker의 action/observation Box 경고다.
Windows에서는 매 실행마다 별도의 새 사용자 임시폴더를 `--basetemp`로 지정했다.
신규 테스트는 합성 오프라인 fixture이며 실제 원천 검증의 대체물이 아니다.
이 표의 종합 failed/종료 코드 1은 알려진 금융 원천 결함을 보존한 결과로,
pytest 실패와 구분한다. KRX의 다른 52개 request JSON 대조는 위 사전 직접
inspection 결과이며, 신규 CLI가 모든 KRX 요청의 공식성을 인증했다고 주장하지 않는다.

```text
python -m black --check src tests dashboard
python -m flake8 src tests dashboard
python -m pytest tests/test_price_evidence.py tests/test_fx_evidence.py tests/test_source_evidence.py -v
python -m pytest -v --basetemp=<new-user-temp>
python -m pytest -m leakage -v --strict-markers --basetemp=<another-new-user-temp>
```

원천·기존 snapshot·기존 감사 결과는 수정하지 않았다. 코드 추가는
`data/price_evidence.py`, `data/fx_evidence.py`, `data/source_evidence.py`와 대응 테스트
3개 및 `config/team12_source_evidence.json`이다. 새 문서 두 개와 기존 수집·스냅샷·
PPO 계약 문서의 설명만 보완했으며 collector/snapshot/audit 구현, 의존성, #94
변환기와 PPO 환경은 변경하지 않았다. Git 커밋·Push·PR 생성은 수행하지 않았다.
