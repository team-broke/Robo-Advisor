"""읽기 전용 학습 모니터.

streamlit run src/robo_advisor/ops/monitor_app.py --server.address 0.0.0.0 \
    --server.port 8501 -- --artifacts-dir artifacts --models-dir models --refresh 10
"""

import argparse
from datetime import datetime, timezone
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from robo_advisor.config import get_settings
from robo_advisor.ops.monitor_data import (
    STALE_SECONDS,
    STATE_LABELS,
    as_dict,
    list_queues,
    list_runs,
    load_run,
    number,
)

METRICS = {
    "cumulative_return": ("누적 수익률", "기간 전체에서 자산 가치가 변한 비율입니다."),
    "mdd": ("MDD", "고점 대비 최대 낙폭입니다. 낮을수록 덜 잃었습니다."),
    "sharpe": ("샤프", "위험 대비 수익입니다. 높을수록 좋으며 기간과 함께 비교하세요."),
}
CONVERGENCE = {
    "converged": "보상 변화가 안정됨",
    "not_converged": "보상 변화가 남아 있음",
    "insufficient_episodes": "아직 판 수가 부족함",
}


def parse_options(argv=None):
    """Streamlit의 -- 뒤에 전달한 경로와 갱신 간격을 해석한다."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="읽기 전용 학습 모니터")
    parser.add_argument("--artifacts-dir", default=str(settings.artifacts_dir))
    parser.add_argument("--models-dir", default=str(settings.models_dir))
    parser.add_argument("--refresh", type=float, default=10)
    options = parser.parse_args(argv)
    if not math.isfinite(options.refresh) or options.refresh <= 0:
        parser.error("--refresh는 0보다 큰 초 단위 숫자여야 합니다.")
    return options


def _format(value, spec=".2f", suffix=""):
    value = number(value)
    return f"{value:{spec}}{suffix}" if value is not None else "기록 없음"


def _duration(value):
    value = number(value)
    if value is None:
        return "기록 없음"
    minutes, seconds = divmod(max(0, int(value)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def _warnings(warnings):
    for warning in warnings:
        st.warning(warning)


def _run_card(row, detail):
    status = detail["status"]
    st.subheader(row["run_id"])
    st.caption(f"{row['status_label']} · 마지막 갱신: {row['updated_at'] or '기록 없음'}")
    if row["stale"]:
        st.warning("상태 갱신이 끊겼습니다. 실행이 멈췄는지는 학습 PC에서 확인하세요.")
    progress = row["progress"]
    st.progress(progress if progress is not None else 0.0)
    if progress is None:
        st.caption("전체 스텝 수가 아직 기록되지 않아 진행률을 계산할 수 없습니다.")
    st.write(
        f"진행 {_format(row['timesteps'], ',.0f')} / "
        f"{_format(row['total_timesteps'], ',.0f')} 스텝"
    )
    left, right = st.columns(2)
    left.metric(
        "초당 스텝",
        _format(status.get("steps_per_second"), ",.0f"),
        help="여러 환경이 수행한 하루 전이의 합산 속도입니다.",
    )
    right.metric(
        "남은 시간", _duration(status.get("eta_seconds")), help="현재 속도로 추정한 시간입니다."
    )
    st.caption(
        f"경과 {_duration(row['elapsed_seconds'])} · 장치 {row['device'] or '기록 없음'}"
        f" · {row['gpu_name'] or 'GPU 정보 없음'}"
    )
    episodes = as_dict(status.get("episodes"))
    left, right = st.columns(2)
    left.metric(
        "최근 판 평균 점수",
        _format(episodes.get("recent_mean_reward"), ".4f"),
        help="최근 종료된 에피소드의 평균 보상입니다. 수익률과 같지는 않습니다.",
    )
    right.metric(
        "최근 판 평균 길이",
        _format(episodes.get("recent_mean_length"), ".0f", "일"),
        help="한 에피소드에서 진행한 평가 관측일 수의 평균입니다.",
    )
    left, right = st.columns(2)
    left.metric(
        "Safe-Guard 종료 비율",
        _format(episodes.get("recent_safeguard_rate"), ".1%"),
        help="최근 판 중 낙폭 한도 초과로 강제 종료된 비율입니다. 기본 한도는 15%입니다.",
    )
    validation = as_dict(status.get("validation"))
    last = as_dict(validation.get("last"))
    right.metric(
        "최근 검증 점수",
        _format(last.get("score"), ".4f"),
        help="학습에 쓰지 않은 검증 기간에서 최근 계산한 선택 지표입니다.",
    )
    st.caption(
        f"검증 기준: {row['validation_metric'] or '미사용'} · "
        f"최적 점수 {_format(row['best_score'], '.4f')} "
        f"({_format(row['best_timesteps'], ',.0f')} 스텝)"
    )
    _warnings(row["warnings"])


def _queues(artifacts_dir):
    queues = list_queues(artifacts_dir)
    if not queues:
        return
    st.header("실험 대기열")
    for queue in queues:
        st.subheader(f"{queue['queue_id']} · {queue['status_label']}")
        st.caption(
            f"대기 {queue['pending']} · 실행 {queue['running']} · 완료 {queue['completed']}"
            f" · 실패 {queue['failed']} · 건너뜀 {queue['skipped']}"
        )
        current = ", ".join(str(value) for value in queue["running_run_ids"]) or "없음"
        st.caption(
            f"현재 실행: {current} · 프로세스 번호: {queue['pid']}"
            f" · 갱신: {queue['updated_at'] or '기록 없음'}"
        )
        if queue["stale"]:
            st.warning("대기열 갱신이 끊겼습니다. 작업 상태는 마지막 기록입니다.")
        jobs = pd.DataFrame(queue["jobs"])
        columns = [
            key for key in ("run_id", "state", "started_at", "finished_at", "error") if key in jobs
        ]
        if columns:
            jobs = jobs[columns].copy()
            if "state" in jobs:
                jobs["state"] = jobs["state"].map(
                    lambda value: (
                        STATE_LABELS.get(value, "확인할 수 없음")
                        if isinstance(value, str)
                        else "확인할 수 없음"
                    )
                )
            st.dataframe(
                jobs.rename(
                    columns={
                        "run_id": "실행 이름",
                        "state": "상태",
                        "started_at": "시작",
                        "finished_at": "종료",
                        "error": "실패 사유",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        _warnings(queue["warnings"])


def _numeric_frame(frame, columns):
    if not set(columns) <= set(frame.columns):
        return pd.DataFrame(columns=columns)
    result = frame[columns].copy()
    for column in columns:
        result[column] = pd.to_numeric(
            result[column].map(lambda value: value if isinstance(value, str) else number(value)),
            errors="coerce",
        )
    return result.replace([float("inf"), -float("inf")], float("nan")).dropna()


def _curves(detail):
    curves = detail["curves"]
    drawn = False
    if not curves.empty:
        for key, title, help_text in (
            ("reward", "한 판 점수", "보상 종류가 다르면 점수 크기를 직접 비교하지 마세요."),
            ("length", "한 판 길이", "한 판에서 진행한 평가 관측일 수입니다."),
            ("safeguard", "Safe-Guard 종료 비율", "낙폭 한도 초과 종료는 1, 그 외는 0입니다."),
        ):
            figure = go.Figure()
            for column, label in ((key, "판별 값"), (f"{key}_smoothed", "최근 평균")):
                values = _numeric_frame(curves, ["timesteps", column])
                if not values.empty:
                    figure.add_trace(
                        go.Scatter(
                            x=values["timesteps"],
                            y=values[column],
                            name=label,
                            mode="lines",
                            opacity=0.35 if column == key else 1,
                        )
                    )
            if figure.data:
                drawn = True
                figure.update_layout(title=title, xaxis_title="누적 스텝", height=260)
                st.plotly_chart(figure, width="stretch")
                st.caption(help_text)
    if not drawn and detail["curve_png"]:
        try:
            st.image(str(detail["curve_png"]), caption="학습곡선", width="stretch")
        except (OSError, ValueError):
            st.warning("학습곡선 이미지를 읽을 수 없습니다.")
    elif not drawn:
        st.info("학습곡선은 첫 판이 끝나고 파일이 만들어지면 표시됩니다.")
    state = detail["convergence"].get("status")
    if isinstance(state, str):
        st.caption(f"수렴 판정: {CONVERGENCE.get(state, state)}. 투자 성과 보장은 아닙니다.")


def _validation(detail):
    history = detail["validation_history"]
    values = _numeric_frame(history, ["timesteps", "score"])
    if values.empty:
        st.info("저장된 검증 이력이 없습니다. 진행 중에는 위의 최근·최적 점수를 확인하세요.")
        return
    figure = go.Figure(
        go.Scatter(x=values["timesteps"], y=values["score"], mode="lines+markers", name="검증 점수")
    )
    validation = as_dict(detail["manifest"].get("validation"))
    best = as_dict(validation.get("best"))
    if number(best.get("timesteps")) is not None and number(best.get("score")) is not None:
        figure.add_trace(
            go.Scatter(
                x=[best["timesteps"]],
                y=[best["score"]],
                mode="markers",
                name="선택한 최적 모델",
                marker={"size": 14, "symbol": "diamond"},
            )
        )
    figure.update_layout(xaxis_title="학습 스텝", yaxis_title="검증 점수", height=320)
    st.plotly_chart(figure, width="stretch")
    st.caption("검증 기간은 모델 선택에 사용합니다. 처음 보는 테스트 기간과 구분하세요.")
    st.dataframe(
        values.rename(columns={"timesteps": "학습 스텝", "score": "검증 점수"}), hide_index=True
    )


def _evaluation(detail):
    periods = detail["evaluation"].get("periods", [])
    if not periods:
        st.info("평가표가 아직 없습니다. evaluation.json이 생기면 여기에 표시됩니다.")
        return
    for label, help_text in METRICS.values():
        st.caption(f"{label}: {help_text}")
    for period in periods:
        st.subheader(str(period.get("label", "평가 기간")))
        st.caption(f"{period.get('start', '시작 미기록')} ~ {period.get('end', '끝 미기록')}")
        rows = period["rows"]
        reference = next((row for row in rows if row.get("strategy") == "equal_weight"), {})
        table = []
        for row in rows:
            strategy = row.get("strategy")
            record = {
                "전략": (
                    {"equal_weight": "균등비중", "best": "검증 최적", "final": "학습 마지막"}.get(
                        strategy, strategy
                    )
                    if isinstance(strategy, str)
                    else "이름 미기록"
                ),
                "누적 수익률": _format(row.get("cumulative_return"), ".2%"),
                "MDD": _format(row.get("mdd"), ".2%"),
                "샤프": _format(row.get("sharpe"), ".3f"),
                "종료 사유": str(row.get("end_reason") or "기록 없음"),
            }
            value = number(row.get("cumulative_return"))
            base = number(reference.get("cumulative_return"))
            delta = value - base if value is not None and base is not None else None
            record["균등비중 대비(%p)"] = _format(
                delta * 100 if delta is not None else None, "+.2f"
            )
            table.append(record)
            if strategy != "equal_weight" and delta is not None:
                st.metric(
                    record["전략"],
                    record["누적 수익률"],
                    f"{delta * 100:+.2f}%p",
                    help="같은 평가 기간의 균등비중 대비 수익률 차이입니다.",
                )
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        st.caption("평가 종료 사유와 기간이 다르면 같은 조건의 성과인지 먼저 확인하세요.")


def _detail(row, detail):
    st.header(f"실행 상세 · {row['run_id']}")
    _warnings(detail["warnings"])
    if row["error"]:
        st.error(str(row["error"]))
    st.caption(
        f"검증 최적 모델: {'있음' if detail['best_model_exists'] else '없음'} · "
        f"마지막 모델: {'있음' if detail['final_model_exists'] else '없음'}"
    )
    charts, validation, evaluation = st.tabs(["학습곡선", "검증 이력", "평가 결과"])
    with charts:
        _curves(detail)
    with validation:
        _validation(detail)
    with evaluation:
        _evaluation(detail)
    manifest = detail["manifest"]
    config = as_dict(manifest.get("config"))
    cutoff = as_dict(manifest.get("data_cutoff"))
    valid = as_dict(manifest.get("validation"))
    env = as_dict(config.get("env"))
    ppo = as_dict(config.get("ppo"))
    with st.expander("주요 설정과 기간"):
        settings = {
            "실제 학습 시작": cutoff.get("actual_first_index"),
            "실제 학습 끝": cutoff.get("actual_last_index"),
            "검증 시작": valid.get("start"),
            "검증 끝": valid.get("end"),
            "목표 스텝": config.get("total_timesteps"),
            "무작위 시작": env.get("random_start"),
            "최소 / 최대 길이": [env.get("min_episode_steps"), env.get("max_episode_steps")],
            "정책망 / 가치망": ppo.get("net_arch"),
            "보상 종류": env.get("reward_variant"),
            "낙폭 벌점 세기": env.get("penalty_lambda"),
            "병렬 환경": manifest.get("n_envs"),
            "무작위 출발값(seed)": manifest.get("seed"),
            "동작 확인용(smoke)": manifest.get("smoke"),
        }
        st.json(settings)
        st.caption("무작위 시작은 판마다 시작 날짜를 바꿉니다. smoke는 본 실험과 구분합니다.")
    with st.expander("최근 실행 로그"):
        st.code(detail["log_tail"] or "아직 로그가 없습니다.", language="text")


def render_monitor(options):
    """주기적으로 모든 표시 영역을 새 산출물로 갱신한다."""
    st.caption(
        f"마지막 화면 갱신 {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} · "
        f"자동 갱신 {options.refresh:g}초 · 응답 없음 기준 {STALE_SECONDS}초"
    )
    runs = list_runs(options.artifacts_dir)
    st.header("지금 학습 중")
    active = [row for row in runs if row["state"] in ("running", "stale")]
    if not active:
        st.info("현재 진행 중으로 기록된 학습이 없습니다.")
    for row in active:
        _run_card(row, load_run(options.artifacts_dir, options.models_dir, row["run_id"]))
    _queues(options.artifacts_dir)
    st.header("실행 기록")
    if not runs:
        st.info("실행 기록이 없습니다. 학습 결과 폴더가 생기면 자동으로 표시됩니다.")
        return
    table = pd.DataFrame(runs)
    columns = {
        "run_id": "실행 이름",
        "status_label": "상태",
        "started_at": "시작(UTC)",
        "finished_at": "종료(UTC)",
        "elapsed_seconds": "소요(초)",
        "timesteps": "스텝",
        "device": "장치",
        "smoke": "동작 확인용",
        "validation_metric": "검증 기준",
        "best_timesteps": "최적 스텝",
        "best_score": "최적 점수",
        "convergence": "수렴 판정",
        "error": "실패 사유",
    }
    st.dataframe(table[list(columns)].rename(columns=columns), hide_index=True, width="stretch")
    selected = st.selectbox("자세히 볼 실행", [row["run_id"] for row in runs], key="monitor_run")
    row = next(row for row in runs if row["run_id"] == selected)
    _warnings(row["warnings"])
    _detail(row, load_run(options.artifacts_dir, options.models_dir, selected))


def main(argv=None):
    options = parse_options(argv)
    st.set_page_config(page_title="학습 모니터", page_icon=None, layout="wide")
    st.title("학습 모니터")
    st.write("학습 진행 상황과 저장된 결과를 확인하세요. 이 화면은 실행 파일을 변경하지 않습니다.")
    st.fragment(run_every=options.refresh)(render_monitor)(options)


if __name__ == "__main__":
    main()
