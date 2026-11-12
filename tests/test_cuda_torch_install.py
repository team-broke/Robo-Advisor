"""설치나 GPU 접속 없이 CUDA 휠 설치 순서와 버전 고정을 검증한다."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import venv

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install_cuda_torch.sh"


@pytest.fixture(scope="module")
def python_venv(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cuda-venv") / "venv with spaces"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(directory)
    return directory


@pytest.fixture
def project(tmp_path, python_venv):
    root = tmp_path / "project with spaces"
    (root / "scripts").mkdir(parents=True)
    script = root / "scripts" / SCRIPT.name
    shutil.copyfile(SCRIPT, script)
    (root / "requirements.txt").write_text("torch==2.14.1+cpu\n", encoding="utf-8")
    env = {**os.environ, "VIRTUAL_ENV": str(python_venv)}
    env.pop("TORCH_CUDA_VARIANT", None)
    return script, env


def _run(project, *args):
    script, env = project
    return subprocess.run(
        ["bash", str(script), *args],
        env=env,
        cwd=script.parent,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("version", ["2.14.1", "2.15.0rc1", "2.14.1.post1"])
def test_dry_run_extracts_locked_version_and_orders_commands(project, version):
    script, env = project
    requirements = script.parents[1] / "requirements.txt"
    requirements.write_text(
        f"# torch==0.0.0+cpu\n--extra-index-url https://download.pytorch.org/whl/cpu\n"
        f"torchvision==0.1.0\n  torch == {version}+cpu  # locked\n",
        encoding="utf-8",
    )
    result = _run(project, "--dry-run")
    assert result.returncode == 0, result.stderr
    first, second = [shlex.split(line) for line in result.stdout.splitlines()[:2]]
    python = str(Path(env["VIRTUAL_ENV"]) / "bin" / "python")
    assert first == [python, "-m", "pip", "install", "-r", str(requirements)]
    assert second == [
        python,
        "-m",
        "pip",
        "install",
        f"torch=={version}+cu130",
        "--index-url",
        "https://download.pytorch.org/whl/cu130",
    ]
    assert "dry-run" in result.stdout


@pytest.mark.parametrize(
    "args,environment,expected",
    [
        ([], "cu128", "cu128"),
        (["--cuda", "cu129"], "cu128", "cu129"),
    ],
)
def test_cuda_index_override(project, args, environment, expected):
    project[1]["TORCH_CUDA_VARIANT"] = environment
    result = _run(project, "--dry-run", *args)
    assert result.returncode == 0, result.stderr
    assert f"https://download.pytorch.org/whl/{expected}" in result.stdout
    assert f"torch==2.14.1+{expected}" in result.stdout


@pytest.mark.parametrize(
    "pin",
    [
        "numpy==1.26.4\n",
        "torch>=2.14\n",
        "torch==2.14.1\n",
        "torch==2.14.1+cpu\ntorch==2.15.0+cpu\n",
        "torch==2.14.1+cu130\n",
    ],
)
def test_missing_ambiguous_or_non_cpu_pin_is_rejected(project, pin):
    (project[0].parents[1] / "requirements.txt").write_text(pin, encoding="utf-8")
    result = _run(project, "--dry-run")
    assert result.returncode != 0
    assert "정확히 하나" in result.stderr
    assert "--index-url" not in result.stdout


@pytest.mark.parametrize(
    "args", [["--cuda"], ["--cuda", "cpu"], ["--cuda", "cu130/other"], ["--unknown"]]
)
def test_invalid_options_fail_before_commands(project, args):
    result = _run(project, "--dry-run", *args)
    assert result.returncode != 0
    assert "pip install" not in result.stdout


def test_active_venv_is_required_even_for_dry_run(project):
    project[1].pop("VIRTUAL_ENV")
    result = _run(project, "--dry-run")
    assert result.returncode != 0
    assert "venv" in result.stderr
    assert not result.stdout


def test_virtual_env_must_point_to_real_venv(project, tmp_path):
    fake = tmp_path / "not-a-venv"
    (fake / "bin").mkdir(parents=True)
    (fake / "bin" / "python").symlink_to(Path(sys.executable).resolve())
    project[1]["VIRTUAL_ENV"] = str(fake)
    result = _run(project, "--dry-run")
    assert result.returncode != 0
    assert "venv" in result.stderr
    assert not result.stdout


def _mock_install(project, tmp_path):
    modules = tmp_path / "modules"
    modules.mkdir()
    log = tmp_path / "calls.jsonl"
    (modules / "pip.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "log = Path(os.environ['INSTALL_TEST_LOG'])\n"
        "with log.open('a') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if len(log.read_text().splitlines()) == int(os.environ.get('FAIL_PIP_CALL', '0')):\n"
        "    sys.exit(17)\n",
        encoding="utf-8",
    )
    (modules / "torch.py").write_text(
        "import json, os\n"
        "with open(os.environ['INSTALL_TEST_LOG'], 'a') as handle:\n"
        "    handle.write(json.dumps('verify_cuda') + '\\n')\n"
        "__version__ = os.environ.get('FAKE_TORCH_VERSION', '2.14.1+cu130')\n"
        "class cuda:\n"
        "    @staticmethod\n"
        "    def is_available():\n"
        "        return os.environ.get('FAKE_CUDA_AVAILABLE', '1') == '1'\n"
        "    @staticmethod\n"
        "    def get_device_name(index):\n"
        "        assert cuda.is_available() and index == 0\n"
        "        return 'Synthetic GPU'\n",
        encoding="utf-8",
    )
    project[1].update(PYTHONPATH=str(modules), INSTALL_TEST_LOG=str(log))
    return log


def test_install_sequence_and_cuda_verification_are_mocked(project, tmp_path):
    log = _mock_install(project, tmp_path)
    result = _run(project)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls == [
        ["install", "-r", str(project[0].parents[1] / "requirements.txt")],
        [
            "install",
            "torch==2.14.1+cu130",
            "--index-url",
            "https://download.pytorch.org/whl/cu130",
        ],
        "verify_cuda",
    ]
    assert "torch version: 2.14.1+cu130" in result.stdout
    assert "torch.cuda.is_available(): True" in result.stdout
    assert "device name: Synthetic GPU" in result.stdout


@pytest.mark.parametrize(
    "variable,value,error",
    [
        ("FAKE_CUDA_AVAILABLE", "0", "CUDA를 사용할 수 없습니다"),
        ("FAKE_TORCH_VERSION", "2.14.1+cpu", "요청한 CUDA 휠 버전과 다릅니다"),
        ("FAKE_TORCH_VERSION", "2.15.0+cu130", "요청한 CUDA 휠 버전과 다릅니다"),
    ],
)
def test_missing_cuda_or_wrong_wheel_exits_nonzero(project, tmp_path, variable, value, error):
    _mock_install(project, tmp_path)
    project[1][variable] = value
    result = _run(project)
    assert result.returncode != 0
    assert error in result.stderr
    assert "torch version:" in result.stdout and "device name:" in result.stdout


@pytest.mark.parametrize("failure", [1, 2])
def test_pip_failure_stops_remaining_steps(project, tmp_path, failure):
    log = _mock_install(project, tmp_path)
    project[1]["FAIL_PIP_CALL"] = str(failure)
    result = _run(project)
    assert result.returncode == 17
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == failure
    assert "verify_cuda" not in calls


def test_dry_run_does_not_install_or_import_torch(project, tmp_path):
    log = _mock_install(project, tmp_path)
    result = _run(project, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert not log.exists()
