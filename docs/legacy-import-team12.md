# 팀용 12자산 원천 자료 가져오기와 #11 보관

이 설정은 사용자가 확정한 투자자산 12종을 별도로 선언한다. 기존 `config/assets.yaml`의 16개 후보, 기존 v5 15자산 실험 설정, v5 원천 자료는 보존한다. 원천 파일을 읽어 #9 `CollectionResult`로 표현하고 #11 스냅샷에 보관하는 단계이며, 저장 성공은 학습 승인이 아니다.

## 확정한 투자자산과 별도 보조 자료

`config/assets_team12.yaml`은 #9 `load_asset_config()`의 schema 1 계약을 따른다. `universe.status`와 12개 ETF의 `selection_status`는 `confirmed`다. 순서는 모델 연결 시 재사용할 명시적 순서이며, 여기서는 모델 입력이나 행동 공간을 생성하지 않는다.

| 순서 | 데이터셋 | 티커 | 로컬 원천 | 2019-09-01 이후 보유 범위·행 수 |
|---:|---|---|---|---|
| 1 | SPY | SPY | Excel `가격원자료` | 2019-09-03–2025-12-31, 1,592행 |
| 2 | IWM | IWM | Excel `가격원자료` | 2019-09-03–2025-12-31, 1,592행 |
| 3 | EFA | EFA | Excel `가격원자료` | 2019-09-03–2025-12-31, 1,592행 |
| 4 | EEM | EEM | Excel `가격원자료` | 2019-09-03–2025-12-31, 1,592행 |
| 5 | IEF | IEF | Excel `가격원자료` | 2019-09-03–2025-12-31, 1,592행 |
| 6 | TIP | TIP | Excel `가격원자료` | 2019-09-03–2025-12-31, 1,592행 |
| 7 | LQD | LQD | Excel `가격원자료` | 2019-09-03–2025-12-31, 1,592행 |
| 8 | GLD | GLD | Excel `가격원자료` | 2019-09-03–2025-12-31, 1,592행 |
| 9 | DBC | DBC | `DBC_2019_2025.csv` | 2019-09-03–2025-12-31, 1,592행 |
| 10 | VNQ | VNQ | Excel `가격원자료` | 2019-09-03–2025-12-31, 1,592행 |
| 11 | KODEX_200TR | 278530.KS | `KODEX_200TR_2019_2025.csv` | 2019-09-02–2025-12-30, 1,528행 |
| 12 | KTB10Y | 148070.KS | `KTB10Y_2019_2025.csv` | 2019-09-02–2025-12-30, 1,528행 |

위 표는 기존 v5 감사에서 확인한 보유 범위다. 실제 가져오기 결과의 행 수와 기간은 요청한 `[start, end)` 범위에 따라 다시 계산한다. 미국 10종의 같은 날짜 수를 국내 ETF에 강제로 적용하지 않는다.

KOSPI(`^KS11`), S&P500(`^GSPC`), Yahoo USD/KRW(`KRW=X`), 금리 후보 DGS3MO는 별도 보조 데이터셋이다. 투자자산 12개에 포함하지 않는다. 이 단계의 일괄 importer는 투자자산 12개만 반환하며, KOSPI CSV는 국내 관측일 차이 비교를 위한 참조로만 읽는다. S&P500·금리·환율을 성공적으로 가져온 것처럼 표시하지 않는다. DGS3MO의 최종 채택 여부·통화·일수익률 환산·일수 계산은 모두 미확정이다.

## 먼저 다루어야 하는 네 가지 문제

### KTB10Y의 실제 OHLC 오류

2020-03-10 원천은 `Open=127475`, `High=127500`, `Low=127500`, `Close=127500`이다. 시가가 저가보다 작으므로 실제 OHLC 범위 오류다. 이 행을 삭제하거나 시가·저가를 고치지 않는다. #10 검증 결과와 원천 행 위치를 남기고 KTB10Y를 `failed`로 보관한다. `Adj Close`가 별도로 존재한다는 이유로 OHLC 오류를 통과시키지 않는다.

향후 공급자 원본 또는 독립 원천으로 정정을 확인하면 별도의 새 스냅샷을 만든다. 이전 스냅샷을 덮어쓰지 않으며, #11의 내용 fingerprint 비교로 변경을 기록한다.

### 국내 ETF의 관측일 차이

감사 대상 기간 `[2019-09-01, 2026-01-01)`에서 두 국내 ETF는 각 1,528행, KOSPI는 1,552행이다. KOSPI CSV에만 있는 관측일 24개를 비교 근거로 남긴다. 이는 공식 KRX 캘린더 대조 결과가 아니므로, 해당 ETF에 반드시 있어야 하는 거래일 24개가 누락되었다고 확정하지 않는다. 날짜를 생성하거나 종가를 전달·보간하지 않으며 `unverified` 사유로 기록한다. KTB10Y의 종합 상태는 실제 OHLC 오류가 우선하므로 계속 `failed`다.

기간을 바꾸면 차이 목록을 그 기간에서 다시 계산한다. 공식 거래소 캘린더·상장 기간·거래정지 이력·공급자 누락 여부를 별도로 확인하기 전에는 미국 거래일 정합이나 KOSPI 관측일 집합을 #12의 거래일 계약으로 대체하지 않는다.

### Yahoo KRW=X의 미확보

기존 `USD_KRW_2019_2025.csv` 등 FRED `DEXKOUS`는 Yahoo `KRW=X`와 다른 원천이다. 이 importer는 DEXKOUS를 `USD_KRW`의 Yahoo 자료로 바꿔 넣지 않는다. `KRW=X` 미확보 사실을 유지한다. 환율의 의미는 1 USD당 KRW이며, 실제 quote 방향 검증·수집·시점 검증은 별도 단계다. 미국 자산을 KRW로 환산하거나 합성 환율을 생성하지 않는다.

### 역사적 공개 시각의 근거 부족

Excel·CSV의 관측 날짜, 파일 수정시각, ZIP 시각, 가져온 현재 시각을 과거 `available_at`으로 사용하지 않는다. CSV는 확인된 수집 기록이 없어 `collected_at=None`이다. Excel은 `이용안내!B3`에 기록된 수집시각을 UTC로 해석할 수 있는 경우 그 값을 기록하되, 원천이 주장한 수집 기록이라는 미검증 사유를 함께 남긴다. 시각을 파싱할 수 없거나 기록이 없으면 `None`으로 둔다. 이 기록도 각 관측값의 역사적 공개시각을 입증하지 않는다. 전 자료의 `available_at=None`, `availability_status=unverified`를 유지한다.

현재 가져온 시각은 `validation.metrics`의 provenance에만 `imported_at`으로 남긴다. #11의 `snapshot_created_at`은 현재 보관 실행 시각이며 원천 수집시각·과거 공개시각과 별개다. 따라서 수치 검사가 통과한 11종도 `unverified`이며, KTB10Y는 `failed`다.

## 원천 선택과 변경 감지

`legacy_import.schema_version=1`은 감사한 로컬 파일을 지정한다. Excel 9종은 동일 workbook SHA-256, CSV 3종은 상대 파일명과 각각의 SHA-256으로 고정한다. 참조 KOSPI CSV도 별도 해시를 둔다. config에는 개인 PC 절대 경로를 넣지 않으며 호출자가 `workbook_path`와 `csv_root`를 전달한다.

파일 바이트 해시가 감사한 값과 다르면 자동으로 신뢰하지 않는다. 변경 또는 잘못된 파일 선택 사실을 명시적으로 오류로 보고한다. 해시 일치는 현재 감사한 파일과 동일하다는 뜻이며, 공급자 데이터의 진실성이나 과거 point-in-time 유효성을 보증하지 않는다.

9개 Excel 자산은 `가격원자료` 시트에서 명시한 `Ticker`와 일치하는 행을 선택한다. 기준100 정규화 시트·모델 입력 파일은 이 경로에 사용하지 않는다. CSV는 `raw_2019_2025` 원천을 사용하며 `model_2020_2025` 파일로 자동 대체하지 않는다. 티커·통화·Source 등 파일 내부 단서가 설정과 충돌하면 검증 오류로 남긴다.

## CollectionResult 계약

`CollectionResult`는 #9 모듈이 정의한 타입이며 #11은 이를 저장한다. 원천을 읽는 행위 자체를 Yahoo/FRED 재수집으로 기록하지 않는다.

- `raw_data`: 선택한 데이터셋의 전체 원천 행과 열을 범위 필터 전에 보존한다. CSV 셀은 빈 문자열을 포함한 문자열로 읽고, Excel은 `가격원자료` 원본 XML에서 해석한 셀 문자열·빈 셀 `None`을 보존한다. `__excel_row`는 원천 행 번호이고 `__excel_cells`는 셀 참조·타입·스타일·XML 표기값을 담은 JSON 문자열이다. 전체 workbook 파일을 뜻하지는 않으며, 다른 티커 행은 해당 자산의 `raw_data`에 섞지 않는다. 바이너리 원본 파일 자체는 별도로 읽기 전용 보존하고 그 SHA-256을 남긴다.
- `data`: 날짜 인덱스와 명시적인 수치 열로 표현한 `[start, end)` 선택 결과다. 파싱 불가능한 날짜는 범위 밖이라는 이유로 조용히 버리지 않고 `NaT`로 남겨 오류를 기록한다. 이 경우 신뢰할 수 없는 실제 기간 끝점은 `None`으로 둔다. 가격·거래량을 조정하거나 결측·중복·역순·OHLC 오류를 제거하지 않는다. 원천 밖 요청 기간을 채우지 않는다.
- `raw_text`: CSV 원문을 보존할 수 있다. Excel 바이트를 텍스트로 위장하지 않는다. 원본 파일은 계속 읽기 전용이며 SHA-256·파일 경로를 provenance로 남긴다.
- `metadata`: 역할·시장·통화·티커, 실제 기간·행 수, 요청한 범위, 가격 정의, 알려지지 않은 수집·공개시각과 종합 상태를 기록한다.
- `validation`: #10 원천 검증 결과와 importer의 자료 출처·파일 SHA-256·선택한 시트·원천 행 위치 등 provenance를 기록한다. #11의 안전한 metadata 허용 목록 밖의 정보는 임의 metadata 확장에만 의존하지 않고 validation의 보존 가능한 metrics에 둔다.

가격 정의는 기존 `Close`와 `Adj Close`를 별개로 유지한다. `Close`를 `Adj Close`로 교체하거나 `Adj Close / Close` 비율로 OHLC를 다시 계산하지 않는다. 감사한 파일에 두 열이 있다는 사실만으로 모든 OHLC에 `auto_adjust=True`가 적용되었다고 주장하지 않는다. `Close == Adj Close`인 국내 TR ETF도 독립적인 배당 조정 검증을 통과한 것으로 처리하지 않는다. 수정주가 근거가 부족하면 `unverified` 사유를 남긴다.

`status=passed`는 필요한 증거에 대해 검증을 통과한 상태다. 이 자료의 선정 확정이나 파일 저장만으로 `passed`로 승격하지 않는다. 실제 오류가 있으면 `failed`, 필요한 독립 증거가 부족하면 `unverified`를 유지한다. `validation.valid`도 저장 성공 여부를 나타내는 용도로 바꾸지 않는다.

## 호출 인터페이스

다음 예시는 프로젝트에 적용한 후의 인터페이스다. 개인 PC 경로는 실행하는 쪽에서 전달한다. 별도 설치한 Parquet 엔진 경로를 프로젝트 코드에 하드코딩하지 않는다.

```python
from robo_advisor.data.importers import import_team12_sources
from robo_advisor.data.snapshot import read_snapshot, write_snapshot

results = import_team12_sources(
    "config/assets_team12.yaml",
    workbook_path=workbook_path,
    csv_root=csv_root,
    start="2019-09-01",
    end="2026-01-01",  # 배타적 종료일
)
snapshot_path = write_snapshot(results, root=snapshot_root)
restored = read_snapshot(snapshot_path)
assert len(restored.datasets) == 12
```

개별 파일을 확인할 때는 `import_legacy_csv(spec, path, *, start, end, expected_sha256=None)` 또는 `import_legacy_workbook(spec, path, *, start, end, expected_sha256=None)`를 사용한다. 팀용 일괄 가져오기는 config의 감사 해시를 사용한다. `expected_sha256=None`인 개별 임의 파일 읽기를 감사 해시 확인 완료로 표시하지 않는다. 지원하지 않는 파일 스키마나 파일 무결성 위반은 `LegacyImportError`로 실패한다. 읽을 수 있는 원천의 실제 가격·OHLC 오류는 `CollectionResult(status=failed)`로 반환하므로 오류 관측을 #11에 보관할 수 있다.

#11 `read_snapshot()`은 파일 SHA-256과 데이터 내용 fingerprint를 검증하고 `datasets`로 원래 `CollectionResult`를 반환한다. #15는 이 인터페이스와 `manifest_sha256`으로 lineage를 고정할 수 있지만, `failed`와 `unverified`를 학습 승인 자료로 자동 사용해서는 안 된다.

## 검증과 작업 범위

Mock CSV·Excel로 티커 선택, 범위 선택, 원천 불변성, OHLC 오류 보존, 누락·중복·역순 보존, 파일 변경 감지, 시점 정보 부재, 12개 설정과 #11 왕복을 확인한다. 감사한 실제 로컬 파일에 대해서도 읽기 전용 import와 로컬 스냅샷 저장·읽기·해시 비교를 수행할 수 있다. 이는 금융 API 다운로드와 별개다.

이번 단계는 #12의 거래일 정합·KRW 환산·해외 종가 래그를 계산하지 않는다. 과거 공개시각·독립 배당 조정·공식 시장 캘린더 검증, Yahoo KRW=X 확보, Rf 정책 합의는 남아 있다. 원천과 기존 실험 설정·팀 모듈·requirements는 수정하지 않는다. Git 커밋·Push·PR 생성도 수행하지 않는다.

실제 실행 결과와 저장 경로, 테스트 통과 개수는 함께 제공하는 최종 검증 보고서를 기준으로 확인한다. staged 패키지에만 작성된 파일은 Desktop 저장소에 적용된 것으로 간주하지 않는다.
