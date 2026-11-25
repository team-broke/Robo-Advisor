"""HTTP 화면 자료만 사용하는 공통 렌더러와 오류·job 상태 표시."""

from datetime import datetime, timezone
from functools import wraps
from io import BytesIO
from urllib.parse import urlsplit
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dashboard.api_client import APIClient, APIError

METRICS = (
    "cumulative_return",
    "cagr",
    "volatility",
    "var_95",
    "cvar_95",
    "mdd",
    "sharpe",
    "sortino",
    "calmar",
    "alpha",
    "beta",
    "information_ratio",
)


def get_client():
    """세션과 비동기 작업의 요청에 같은 상관 ID를 전달한다."""
    if "correlation_id" not in st.session_state:
        st.session_state.correlation_id = str(uuid4())
    return APIClient(correlation_id=st.session_state.correlation_id)


def show_error(error):
    st.warning(error.user_message)
    # 원문 오류 메시지에는 서버 경로나 비밀값이 들어 있을 수 있다.
    st.text(f"추적 ID: {error.correlation_id}")


def page(title):
    def decorate(render):
        @wraps(render)
        def wrapped():
            st.set_page_config(page_title=f"{title} · Robo-Advisor", layout="wide")
            st.title(title)
            try:
                with get_client() as client:
                    render(client)
            except APIError as error:
                show_error(error)
            except (ValueError, TypeError, KeyError, AttributeError):
                st.warning("화면 자료 형식 또는 API 주소 설정을 확인하세요.")

        return wrapped

    return decorate


def section(client, name):
    response = client.section(name)
    provenance = response.get("provenance")
    if provenance and provenance["kind"] == "synthetic":
        st.warning("합성 데이터 기반 시스템 검증 결과입니다. 실제 투자 성과가 검증되지 않았습니다.")
        st.caption(f"실행 {provenance['run_id']} · 후보 평가 정책 · 학습 승인 보류")
    elif provenance and provenance["status"] == "unverified":
        st.warning("이 연구 결과는 검증 및 학습 승인 대기 상태입니다.")
    st.caption(f"평가 시각 {response['as_of']} · 산출물 {response['artifact_version'][:12]}")
    if response.get("model_id"):
        st.caption(f"모델 {response['model_id']}")
    return response


def series_chart(points, *, title, index="timestamp"):
    st.subheader(title)
    if not points:
        st.info("저장된 시계열이 없습니다.")
        return
    frame = pd.DataFrame(points)
    if index == "timestamp":
        frame[index] = pd.to_datetime(frame[index], utc=True)
    frame = frame.pivot(index=index, columns="series", values="value").sort_index()
    st.line_chart(frame)


def saved_plots(client, response, *, role=None):
    plots = [item for item in response.get("plots", []) if role is None or item["role"] == role]
    for plot in plots:
        st.image(BytesIO(client.plot(response["section"], plot["id"])), caption=plot["title"])
    return bool(plots)


@page("Robo-Advisor")
def render_home(client):
    st.write("사전 계산된 자산배분·성과·설명·리스크와 출처가 있는 리서치를 조회합니다.")
    status = client.health()
    st.subheader("API 준비 상태")
    services = status.get("services", {})
    if not services:
        st.info("등록된 서비스가 없습니다.")
    for name, ready in services.items():
        if ready:
            st.success(f"{name}: 준비 완료")
        else:
            st.warning(f"{name}: 미준비 · {status.get('reasons', {}).get(name, 'not_configured')}")
    st.caption("왼쪽 메뉴에서 화면을 선택하세요. API_BASE_URL 환경변수로 서버를 지정합니다.")


@page("포트폴리오")
def render_portfolio(client):
    data = section(client, "portfolio")["data"]
    st.subheader("최신 비중")
    weights = data.get("weights")
    if weights is None:
        st.info("저장된 포트폴리오 비중이 없습니다.")
    else:
        rows = weights["asset_weights"] + [{"asset": "현금", "weight": weights["cash_weight"]}]
        frame = pd.DataFrame(rows).set_index("asset")
        st.bar_chart(frame)
        st.dataframe(frame.style.format("{:.2%}"))
    series_chart(data.get("cumulative_returns", []), title="누적 수익률")
    st.caption("수익률은 소수 단위입니다(0.10 = 10%).")


@page("RL 성과")
def render_performance(client):
    data = section(client, "rl_performance")["data"]
    series_chart(data.get("learning_curve", []), title="학습 곡선", index="step")
    st.subheader("12개 성과 지표")
    metrics = data.get("metrics", [])
    if metrics:
        frame = pd.DataFrame({item["strategy"]: item["metrics"] for item in metrics}).reindex(
            METRICS
        )
        st.dataframe(frame.style.format("{:.4f}", na_rep="—"))
        if frame.isna().any().any():
            st.info("— 항목은 API에 저장된 값이 없습니다.")
    else:
        st.info("저장된 성과 지표가 없습니다.")
    series_chart(data.get("comparison", []), title="DRL · 벤치마크 · 동일가중 · MVO 비교")
    st.caption("비교선은 API가 제공하는 전략만 표시합니다. 수익률은 소수 단위입니다.")


@page("SHAP 의사결정 설명")
def render_shap(client):
    response = section(client, "shap")
    data = response["data"]
    st.caption(f"출력 {data.get('output') or '미지정'} · 단위 {data.get('unit', '미지정')}")
    st.subheader("Summary Plot")
    if not saved_plots(client, response, role="summary"):
        if data.get("summary"):
            st.bar_chart(pd.DataFrame(data["summary"]).set_index("feature")["mean_abs_shap"])
        else:
            st.info("저장된 Summary 자료가 없습니다.")
    st.subheader("Force Plot")
    if not saved_plots(client, response, role="force"):
        if data.get("force") and data.get("base_value") is not None:
            force_plot(data)
        else:
            st.info("저장된 Force 자료가 없습니다.")
    st.caption("기여도는 API가 저장한 모델 출력 단위이며, 포트폴리오 비중 변화량이 아닙니다.")


def force_plot(data):
    """저장된 기여도와 기준값을 누적 막대로 배치한다. 설명 계산은 하지 않는다."""
    rows = data["force"]
    figure, axis = plt.subplots(figsize=(9, max(3, len(rows) * 0.35)))
    position = data["base_value"]
    for index, row in enumerate(rows):
        value = row["contribution"]
        axis.barh(
            index,
            abs(value),
            left=min(position, position + value),
            color="#B8423A" if value > 0 else "#3277A8",
        )
        position += value
    axis.set_yticks(range(len(rows)), [row["feature"] for row in rows])
    axis.axvline(data["base_value"], color="#777777", linestyle="--", label="base")
    if data.get("prediction") is not None:
        axis.axvline(data["prediction"], color="#222222", label="prediction")
    axis.set_xlabel(data.get("unit", "SHAP contribution"))
    axis.legend()
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


def source_link(label, url):
    parts = urlsplit(str(url))
    if parts.scheme in {"http", "https"} and parts.hostname and not parts.username:
        st.link_button(label, str(url))
    else:
        st.caption("출처 URL 형식을 확인할 수 없습니다.")


def research_result(result):
    st.subheader("공개 실행 로그")
    st.caption(f"Trace ID: {result.get('trace_id', '미제공')} · 재시도 {result.get('retries', 0)}")
    is_body = result.get("evidence_scope") == "dart_body"
    if result.get("evidence_scope") == "dart_listing_metadata":
        st.info(
            "실제 OpenDART 공시 목록의 제목과 접수일을 확인했습니다. 공시 본문 분석은 포함되지 않습니다."
        )
    elif is_body:
        st.info("OpenDART 공시 본문 근거 · 검증된 원문 발췌와 원문 링크를 표시합니다.")
    trace = result.get("trace", [])
    if trace:
        keys = ("step", "tool", "query", "source_ids", "validation", "retries", "elapsed_ms")
        st.dataframe(pd.DataFrame([{key: row.get(key) for key in keys} for row in trace]))
    else:
        st.info("공개 실행 로그가 없습니다.")
    if result.get("status") != "completed":
        st.warning("근거 검증을 통과하지 못했습니다. 검증된 리서치 결과가 없습니다.")
        return
    st.subheader("리서치 결과")
    claims = result.get("claims", [])
    if not claims:
        st.info("인용이 있는 결과가 없습니다.")
    for claim in claims:
        if not claim.get("citations"):
            continue
        if is_body:
            st.text(claim["text"])
        else:
            st.write(claim["text"])
        for citation in claim["citations"]:
            body = citation.get("body")
            if body:
                st.caption(f"{body['section']} · {body['member']} · {body['locator']}")
                st.caption(
                    f"접수일 {body['receipt_date']} (시각 미확인) · 사용 가능 {citation['available_at']}"
                )
                source_link(f"DART 원문 {body['receipt']}", citation["url"])
            else:
                source_link(
                    f"{citation['doc_id']} · 공개 {citation['published_at']}", citation["url"]
                )
    if result.get("risk_candidates"):
        st.caption("위험 이벤트 후보는 점수를 산정하지 않으며 PPO에 반영하지 않습니다.")
        for candidate in result["risk_candidates"]:
            st.text(f"{candidate['event_type']} · {candidate['source_id']}")


@st.fragment(run_every="2s")
def research_progress():
    """한 번씩 상태를 조회한다. rerun·네트워크 오류가 작업을 재제출하지 않는다."""
    job_id = st.session_state.get("research_job")
    if job_id:
        st.caption(f"Job ID: {job_id}")
        st.button("상태 새로고침", key="refresh_job")
        try:
            with get_client() as client:
                job = client.research_job(job_id)
            status = job["status"]
            if status in {"queued", "running"}:
                st.info("대기 중" if status == "queued" else "리서치 실행 중")
                st.caption("완료 후 공개 단계 로그가 표시됩니다. 2초마다 상태를 조회합니다.")
            elif status == "completed":
                st.session_state.research_result = job["result"]
                del st.session_state.research_job
                st.rerun()
            elif status == "failed":
                st.session_state.research_job_failed = True
                del st.session_state.research_job
                st.rerun()
        except APIError as error:
            show_error(error)
        except (ValueError, TypeError, KeyError):
            st.warning("작업 상태 응답 형식을 확인하세요.")
    if st.session_state.get("research_job_failed"):
        st.warning("리서치 작업이 실패했습니다. Job ID와 추적 ID로 서버 상태를 확인하세요.")
    if st.session_state.get("research_result"):
        try:
            research_result(st.session_state.research_result)
        except (ValueError, TypeError, KeyError, AttributeError):
            st.warning("리서치 결과 응답 형식을 확인하세요.")
    elif not job_id and not st.session_state.get("research_job_failed"):
        st.info("질문을 제출하면 작업 상태와 인용이 있는 결과가 표시됩니다.")


@page("출처 기반 리서치")
def render_research(client):
    st.caption("공개 실행 로그는 단계·도구·질의·출처·검증 메타데이터만 표시합니다.")
    if "research_as_of" not in st.session_state:
        st.session_state.research_as_of = datetime.now(timezone.utc).isoformat()
    with st.form("research_query"):
        question = st.text_area("질문 또는 티커")
        as_of = st.text_input("평가 시각 (시간대 포함)", key="research_as_of")
        submitted = st.form_submit_button(
            "리서치 요청", disabled=bool(st.session_state.get("research_job"))
        )
    if submitted:
        try:
            stamp = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
            if stamp.tzinfo is None or stamp.utcoffset() is None or not question.strip():
                raise ValueError
        except ValueError:
            st.warning("질문과 시간대가 포함된 평가 시각을 입력하세요.")
        else:
            response = client.research(question.strip(), stamp.isoformat())
            st.session_state.pop("research_result", None)
            st.session_state.pop("research_job_failed", None)
            st.session_state.research_last_job = response.data.get("job_id")
            if response.status_code == 202:
                st.session_state.research_job = response.data["job_id"]
                st.rerun()
            else:
                st.session_state.research_result = response.data
    research_progress()
    if st.session_state.get("research_last_job"):
        st.caption(f"최근 Job ID: {st.session_state.research_last_job}")
    if st.session_state.get("correlation_id"):
        st.caption(f"추적 ID: {st.session_state.correlation_id}")
    render_news(client)


def render_news(client):
    st.divider()
    st.subheader("네이버 검색 · 금융 뉴스")
    with st.form("naver_news_query"):
        query = st.text_input("금융 뉴스 검색어", value="금융시장 ETF", max_chars=100)
        submitted = st.form_submit_button("네이버 뉴스 검색")
    if not submitted:
        return
    if not query.strip():
        st.warning("검색어를 입력하세요.")
        return
    try:
        items = client.news(query.strip())["items"]
    except APIError as error:
        show_error(error)
        return
    if not items:
        st.info("검색 결과가 없습니다.")
    else:
        st.caption(f"네이버 검색 결과 {len(items)}건 · 원문 링크 제공")
    for item in items:
        st.text(item["title"])
        st.text(item["summary"])
        stamp = datetime.fromisoformat(item["published_at"].replace("Z", "+00:00"))
        st.caption(f"게시 {stamp:%Y-%m-%d %H:%M %Z}")
        source_link("기사 원문" if item["original_url"] else "네이버 검색 결과 링크", item["url"])


@page("ANOVA 검정")
def render_anova(client):
    response = section(client, "anova")
    data = response["data"]
    st.caption(f"주 지표 {data.get('metric') or '미지정'} · alpha {data.get('alpha', .05)}")
    for key, title in (
        ("results", "효과별 F · p · eta²"),
        ("diagnostics", "가정 진단"),
        ("tukey", "조건부 Tukey HSD"),
    ):
        st.subheader(title)
        if data.get(key):
            st.dataframe(pd.DataFrame(data[key]))
        else:
            st.info(
                "사후검정이 수행되지 않았거나 저장된 자료가 없습니다."
                if key == "tukey"
                else "저장된 검정 자료가 없습니다."
            )
    saved_plots(client, response)


@page("리스크 모니터링")
def render_risk(client):
    data = section(client, "risk_monitor")["data"]
    for column, key, title in zip(st.columns(2), ("var", "cvar"), ("VaR 95%", "CVaR 95%")):
        with column:
            value, limit = data.get(f"{key}_95"), data.get(f"{key}_limit")
            st.metric(title, "미제공" if value is None else f"{value:.2%}")
            if value is not None and limit is not None:
                st.progress(min(1.0, max(0.0, value / limit)), text=f"한도 {limit:.2%}")
                if value > limit:
                    st.warning("설정 한도 초과")
            else:
                st.info("저장된 위험값과 한도가 있어야 게이지를 표시합니다.")
    series_chart(data.get("drawdown", []), title="낙폭 추이 (MDD 모니터링)")
    st.caption("낙폭은 소수 단위입니다. 한도와 위험값은 API의 저장된 값을 사용합니다.")
    st.subheader("Safe-Guard 알림")
    events = data.get("safeguard_events", [])
    for event in events:
        st.warning(f"{event['timestamp']} · {event['reason']} · Trace {event['trace_id']}")
    if not events:
        st.info("저장된 Safe-Guard 이벤트가 없습니다.")
    st.subheader("리스크 태그 출처")
    tags = data.get("tags", [])
    for tag in tags:
        st.write(
            f"{tag['asset']} · {tag['event_type']} · "
            f"심각도 {tag['severity']:.2f} · 신뢰도 {tag['confidence']:.2f}"
        )
        st.caption(
            f"공개 {tag['published_at']} · 가용 {tag['available_at']} · Trace {tag['trace_id']}"
        )
        source_link(f"{tag['asset']} 출처", tag["source_url"])
    if not tags:
        st.info("저장된 리스크 태그가 없습니다.")
