"""MDD 한도 경계·최초 초과 시점·고점 갱신을 확인한다."""

import logging

import pytest
from pydantic import ValidationError

from robo_advisor.rl.safeguard import MDDSafeguard


@pytest.mark.parametrize("initial_nav", [1.0, 100.0, 123.45])
def test_exact_fifteen_percent_does_not_terminate(initial_nav):
    guard = MDDSafeguard(initial_nav, "trace-boundary")
    result = guard.update(initial_nav * 0.85)

    assert result.drawdown == 0.15
    assert not result.terminated
    assert result.reason is None
    assert guard.termination_result is None


def test_drawdown_just_above_limit_terminates():
    guard = MDDSafeguard(100, "trace-exceeded")
    result = guard.update(100 * (1 - 0.1500001))

    assert result.terminated
    assert result.reason == "mdd_limit_exceeded"
    assert result.step_index == 0
    assert result.peak_nav == 100
    assert result.nav == pytest.approx(84.99999)
    assert result.drawdown == pytest.approx(0.1500001)
    assert result.trace_id == "trace-exceeded"
    assert guard.termination_result is result


def test_only_first_exceeded_step_is_reported_and_logged(caplog):
    guard = MDDSafeguard(100, "trace-first")
    with caplog.at_level(logging.INFO, logger="robo_advisor.rl.safeguard"):
        results = [guard.update(nav) for nav in [100, 90, 85, 84, 80, 110, 80]]

    assert [result.step_index for result in results if result.terminated] == [3]
    assert guard.termination_result is results[3]
    assert guard.termination_result.peak_nav == 100
    assert guard.termination_result.drawdown == pytest.approx(0.16)
    assert all(result.reason is None for result in results[4:])
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.trace_id == "trace-first"
    assert record.step_index == 3
    assert record.reason == "mdd_limit_exceeded"
    assert record.peak_nav == 100
    assert record.drawdown == pytest.approx(0.16)


def test_high_water_mark_updates_and_current_drawdown_recovers():
    guard = MDDSafeguard(100, "trace-peak")
    assert guard.update(90).drawdown == pytest.approx(0.1)
    high = guard.update(120)
    assert high.peak_nav == 120
    assert high.drawdown == 0
    boundary = guard.update(102)
    assert boundary.drawdown == 0.15
    assert not boundary.terminated
    recovered = guard.update(114)
    assert recovered.drawdown == pytest.approx(0.05)
    result = guard.update(101)
    assert result.terminated
    assert result.peak_nav == 120
    assert result.drawdown == pytest.approx(19 / 120)
    assert result.step_index == 4


def test_custom_limit_and_trimmed_trace_id():
    guard = MDDSafeguard(100, "  trace-custom  ", limit=0.2)
    assert not guard.update(80).terminated
    result = guard.update(79)

    assert result.terminated
    assert result.trace_id == "trace-custom"
    assert result.drawdown == pytest.approx(0.21)


@pytest.mark.parametrize("nav", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_invalid_nav_rejected_without_advancing_state(nav):
    with pytest.raises(ValueError):
        MDDSafeguard(nav, "trace-invalid")
    guard = MDDSafeguard(100, "trace-invalid")
    with pytest.raises(ValueError):
        guard.update(nav)
    result = guard.update(84)

    assert result.step_index == 0
    assert result.peak_nav == 100
    assert result.terminated


@pytest.mark.parametrize("limit", [0, -0.1, 1.01, float("nan"), float("inf")])
def test_invalid_limits_rejected(limit):
    with pytest.raises(ValueError):
        MDDSafeguard(100, "trace-limit", limit=limit)


@pytest.mark.parametrize("trace_id", ["", "   ", None, 123])
def test_common_trace_id_contract_is_enforced(trace_id):
    with pytest.raises(ValidationError):
        MDDSafeguard(100, trace_id)


def test_full_drawdown_limit_does_not_terminate_positive_nav():
    guard = MDDSafeguard(100, "trace-full", limit=1)

    assert not guard.update(0.001).terminated
