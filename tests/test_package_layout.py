"""패키지 골격이 import 가능한지 확인한다."""

import importlib

import pytest

MODULES = [
    "robo_advisor",
    "robo_advisor.schemas",
    "robo_advisor.data",
    "robo_advisor.features",
    "robo_advisor.rl",
    "robo_advisor.backtest",
    "robo_advisor.baselines",
    "robo_advisor.research",
    "robo_advisor.explain",
    "robo_advisor.experiments",
    "robo_advisor.integration",
    "robo_advisor.api",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_importable(name):
    assert importlib.import_module(name) is not None


def test_version_exposed():
    import robo_advisor

    assert robo_advisor.__version__
