# OpenDART 공시 본문 리서치

OpenDART의 `GET /api/document.xml`에 등록된 인증키와 공시 목록의 14자리
접수번호를 전달한다. 정상 응답은 ZIP 바이너리이며, 공시가 없거나 호출 제한
등에 걸리면 OpenDART 상태 코드가 포함된 XML이 올 수 있다. 브라우저의 공시 원문
링크는 인증 정보가 없는 DART 뷰어 주소를 사용한다.
[OpenDART 원본파일 개발가이드](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019003)

금융감독원은 오픈API에서 내려받은 공시 원문을 활용할 수 있다고 안내한다.
사용에는 등록된 인증키와 서비스 약관이 적용되며, 계정별 호출 제한은 다를 수 있다.
[오픈API 활용 안내](https://opendart.fss.or.kr/intro/main.do) ·
[OpenDART 이용약관](https://opendart.fss.or.kr/intro/terms.do)

## 수집 및 실행

공시 목록은 `scripts/prepare_dart_research.py`로 준비한다. 본문 수집기는 목록
manifest의 신뢰 SHA-256과 목록 파일의 SHA-256을 확인한 뒤, 지정된 접수번호 최대
3건만 요청한다. 같은 출력 폴더의 ZIP과 최초 수신 기록을 재사용한다. 수집이 중단된
접수번호는 중복 요청을 막기 위해 자동 재시도하지 않는다.

```powershell
$env:PYTHONPATH = 'src'
python scripts/prepare_dart_body.py --env-file <ignored-env-file> `
  --listing <listing-manifest> --listing-sha256 <trusted-sha256> `
  --output artifacts/live-dart-body --receipts 20261116001482 20261113001338 20261113801935
```

`--acquire-only`로 원본만 수집하고, 이후 `--build-only`로 네트워크 없이 corpus를
만들 수 있다. 원본 ZIP과 수신 기록, `documents.jsonl`, `chunks.jsonl`,
`research_manifest.json`은 Git이 무시하는 `artifacts/`에 둔다. 컨테이너에서는
`API_RESEARCH_MANIFEST`에 읽기 전용 마운트의 manifest 경로,
`API_RESEARCH_MANIFEST_SHA256`에 신뢰 해시를 지정한다. 기존 API와 Streamlit을
띄운 뒤 수집 시각 이후의 `as_of`로 `POST /research`를 요청한다. 배포에 지정된
manifest가 `dart_body` 또는 `dart_listing_metadata` 중 서비스할 근거 범위를
결정한다. Streamlit은 응답에 표시된 범위에 따라 본문 근거와 목록 제목·접수일을
구분한다.

## 근거와 시점

- 목록 문서의 300자 요약은 기존 계약을 유지한다. 별도 본문 청크에는 원문 파일명,
  본문 요소 및 행 번호, 절 제목, 기간·단위·발췌문, XML과 발췌문 각각의 SHA-256을
  둔다. 요소 번호는 `BODY`를 순회한 순서이며, 표 행 번호는 헤더를 포함한다.
  표의 셀은 ` | `로 연결하며 발췌문에 필요한 원문 단위와 헤더를 함께 표시한다.
- ZIP은 디스크에 풀지 않는다. 크기, 확장 비율, 파일 수를 제한하고 절대 경로,
  경로 탈출, 링크, 중복 파일명을 차단한다. XML은 DTD와 엔터티 선언을 거부한다.
  실제 DART XML에서 확인한 리터럴 `&`와 `< … >` 표기만 파싱용으로 이스케이프하며,
  변환 횟수와 수정 전 XML 해시를 기록한다. HTML로 감싼 문서는 제외 사유를 남긴다.
- OpenDART 목록은 접수일 단위다. 접수 시각은 미확인으로 기록한다. 본문 가용 시각은
  다음 날 0시(한국 시간)와 최초 본문 수신 시각 중 늦은 시각이다. ChromaDB와
  LangGraph가 각각 평가 시각을 확인한다.
  [OpenDART 공시검색 개발가이드](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019001)
- 목록에서 원본에 후속 정정·철회 표시가 있거나 정정·첨부 추가 문서의 버전 관계를
  확인할 수 없으면 원본 ZIP과 사유를 보존하고 검색에서 제외한다. 초기 수집 후
  새로 제출되는 정정을 자동 추적하거나 정정 전후의 전체 이력을 복원하는 동작은
  구현 범위에서 제외했다.
- FastAPI가 manifest, ZIP, 수신 기록, 청크 해시를 확인하고 원문에서 청크를
  재생성해 제출본과 대조한다. 본문 답변에서 Gemini는 질의를 구성하고 근거별
  관련성을 판정한다. LangGraph의 마지막 단계는 서버에서 검증된 발췌문과 원문 URL을
  조합한다. 본문 질문 1건당 Gemini는 최대 2회 호출되며, 근거 바깥의
  인용이나 임의 문장·수치가 포함된 모델 응답은 거부한다.
- 명시적인 실적 쇼크 또는 규제 변경 문구가 인용됐을 때 기존 분류기로 위험 이벤트
  후보를 기록할 수 있다. 심각도, 신뢰도, 자산 매핑을 계산하지 않는다. 본문
  후보는 RiskTag 변환에서 제외되고, `training_approved=false`를 유지한다.
  네이버 뉴스는 기존 요청형 조회를 그대로 사용한다.

## 2026-11-25 로컬 실증

| 접수번호 | 공시 | 본문 처리 |
|---|---|---|
| 20261116001482 | 삼성전자 분기보고서 (2026.09) | XML 검증 후 10개 청크 색인 |
| 20261113001338 | 동일인 등 출자 계열회사와의 상품·용역거래 변경 | XML 검증 후 1개 청크 색인 |
| 20261113801935 | 최대주주 등 소유주식변동신고서 | ZIP과 XML 멤버 해시 기록, HTML 형식으로 색인 제외 |

`gemini-3.5-flash-lite`로 삼성전자 분기보고서의 2026년 3분기 연결 3개월
매출액과 영업이익을 조회했다. 인용한 두 행과 SHA-256이 로컬 청크와 일치했고
Streamlit에서 행별 원문 링크를 확인했다. 본문 증빙의 최초 가용 시각 직전에는
ChromaDB 검색 결과가 없었다. 최종 trace ID는 `e092fdf5-d1dc-4ebb-a6b7-865b49337b96`이다.
