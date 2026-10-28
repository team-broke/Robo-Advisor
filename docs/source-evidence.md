# #9·#11 후속: 원천 증거 sidecar

보존한 원천의 해시·수집 영수증·값·빈티지 연결을 보완한다. 기존 실제 데이터 감사
`audit_research_package`를 재사용하며, 가격·환율·JSON·snapshot을 수정하거나
공급자 값을 새로 내려받지 않는다. #94 변환기나 PPO 학습은 실행하지 않는다.

## 한 번의 실행과 결과 해석

기존 의존성과 프로젝트 패키지가 설치된 저장소 루트에서 실행한다.

```powershell
python -m robo_advisor.data.source_evidence --config config/team12_source_evidence.json --root '<evidence-root>' --output '<new-report>.json'
```

`<evidence-root>`는 보존한 `team12-experimental-<delivery-id>` 패키지 루트다.
설정은 상대 경로와 고정 SHA-256을 사용한다. 출력 부모 폴더를 먼저 준비하고
새 파일명을 지정한다. 원본·기존 보고서는 덮어쓰지 않는다.
종료 코드는 `0=통과`, `1=결함/입력 오류`, `2=미확인`이며 어떤 결과도 학습 승인은 아니다.

부모 snapshot의 KTB10Y·Yahoo USD_KRW `failed`, 연구 선택 원천의 `unverified`,
`actual_available_at=null`과 `training_approved=false`를 유지한다. 종합 `failed`는
알려진 결함을 보존한 검출 결과이며 수치 재계산 성공으로 이를 통과시키지 않는다.

## 실제 보존 원문과 연결하는 범위

| 대상 | 보존 원문 대응 | 해석과 한계 |
|---|---|---|
| SMBS | CSV 1,561행을 선택 HTML 원문 7개와 대조 | 원문 값 일치와 수집 receipt 검사. HTTP 전송 인증·역사적 공개 검증은 아님 |
| ALFRED | CSV 1,599행 중 null 20개를 포함해 선택 빈티지 원문 331개와 대조 | 요청 series·빈티지·날짜·값을 비교. null은 그대로 보존 |
| SMBS 선택 외 자료 | receipt는 있으나 원문이 없는 optional 응답 3개 기록 | 부분 보존이며 전체 원문 확보라고 주장하지 않음 |
| KRX 후보 | KTB10Y 단일 후보의 OHLCV·보존 CSV 대조, 국내 부재일은 참조 달력과 대조 | 일부 날짜·필드의 근거이며 전 기간 가격 기준·조정 이력·채택 승인과 별개 |
| Yahoo KRW=X | 원 JSON → raw frame → 요청 범위 OHLCV를 대조 | 오류가 보존 JSON에도 있는지 확인. 공급자 내부 원인·휴일 정책까지 확정하지 않음 |

해시·바이트 수·요청 URL/series/vintage·HTTP 상태와 원문 값을 함께 검사한다.
선택된 FX 원문에는 정상 수집 receipt가 필요하다. 선택 외 실패 요청이나 optional
원문 부재도 미확인 사유로 남긴다. 같은 값이 다른 원천에 있다고 자동 대체하지 않는다.

## 공식 문서 확인: 2026-10-28 KST

**새 웹 조회와 기존 원문 보존 검사를 구분한다.** 아래 정의·일정 확인은
2019~2025 모든 행의 실제 공개시각을 확인한 결과가 아니다.

- [FRED DEXKOUS](https://fred.stlouisfed.org/series/DEXKOUS): 1 USD당 KRW,
  일별, 뉴욕 정오의 전신환 매입률이다. 정오 fixing은 관측 기준이며 공개시각이 아니다.
- [H.10 설명](https://www.federalreserve.gov/releases/h10/about.htm)과
  [공식 발표·정정 이력](https://www.federalreserve.gov/feeds/h10.html): 전주 일별 환율의
  주간 발표이며 일반 일정은 월요일 16:15 ET다. 과거 release와 이후 수정본이
  다를 수 있고 실제 지연 사례도 있다. 일정표를 개별 실제 release 로그로 쓰지 않는다.
- [ALFRED 관측 API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html#vintage_dates):
  `vintage_dates`는 특정 과거 날짜의 상태를 조회하는 YYYY-MM-DD이며 실제 변경일이
  아닌 임의 날짜도 지정할 수 있다. [빈티지 날짜 API](https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html)는
  새 값·개정으로 변경된 발표 날짜의 목록을 별도로 제공한다.
- [ALFRED Help](https://alfred.stlouisfed.org/help): 통상 발표 후 1영업일 내에 적재한다.
  원천 발표 날짜가 없으면 제공자 날짜, 그것도 없으면 FRED 최초 이용 가능 날짜를
  쓸 수 있다. [real-time period](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html)는
  날짜 단위이므로 그 자체가 정확한 일중 시각·시간대 증거는 아니다.

기존 연구의 **수요일 빈티지**는 조회·샘플링 정책이며 공식 DEXKOUS 발표요일이
아니다. **다음날 00:00 New York부터 사용**하는 것도 `assumed_usable_at`이다.
고정 UTC 시각이나 실제 공개시각으로 바꾸지 않는다. 관측일·빈티지·발표·수집을
분리하고, 확인된 actual이 있으면 이른 가정보다 우선한다.

### SMBS와 보존 HTTP 기록

이번 새 웹 조회에서 [공식 FAQ](http://www.smbs.biz/Customer/CusMain.jsp?id=4),
[기간표](http://www.smbs.biz/ExRate/StdExRate.jsp),
[오늘의 환율](http://www.smbs.biz/ExRate/TodayExRate.jsp)은 HTTPS 접근이 타임아웃됐다.
새 공식 웹 확인을 완료했다고 표시하지 않는다.

2026-10-17 보존 FAQ는 현물환 거래량 가중평균 MAR 및 영업일 08:30경 고시를
안내한다. 이 보존본은 receipt와 SHA-256이 일치하지만 HTTP 수집이다.
SMBS의 고시일, MAR 산출 거래 구간, 실제 공개·수집시각은 구분해야 한다.
MAR를 당일 종가나 뉴욕 정오 환율로 부르지 않는다. 현재의 일반 고시 일정이나
해시 일치가 모든 역사적 행의 실제 공개·수정 이력을 증명하지 않는다.

보존 `H10-about.html`의 receipt는 HTTP 403이다. 이를 정상 H.10 설명 원문으로
채택하지 않는다. 오늘 공식 H.10 문서에 접근한 사실과 과거 실패 기록은 별개이며,
실패 원문·receipt를 새 성공 응답으로 바꾸지 않는다.

## #11 저장과 PPO 연결 게이트

#11 metadata는 허용 필드를 검사한다. 임의 증거 필드를 metadata 최상위에 넣거나
기존 manifest를 수정하지 않는다. 이번 증거는 source·manifest SHA-256에 연결한
별도 sidecar로 보관한다. 향후 새 snapshot 버전을 합의하면 허용된 `validation`
하위의 유한 JSON에 근거를 연결할 수 있으며 비밀정보와 개인 경로를 넣지 않는다.

KTB10Y의 부모 Open < Low 결함은 그대로다. 후보 Low나 일부 KRX 관측의 일치만으로
Yahoo와 같은 조정 기준·분배금/분할 이력·전 기간 공식 현물 일정이 확인되는 것은
아니다. 정정 CSV의 독립 수집 receipt가 없으면 다른 날짜의 근거로 추정하지 않는다.

[기존 실제 감사](data-audit.md), [실측 인벤토리](real-data-inventory.md),
[PPO 입력 계약](ppo-data-readiness.md)과 함께 검토한다. 실제 역사적 공개,
가격 조정·총수익 기준, 공식 전체 일정, 연속 구간·Window 경계 및 체결·환전·비용과
#94 변환기는 남아 있다. 원문 대응의 성공은 학습 승인이 아니다.
