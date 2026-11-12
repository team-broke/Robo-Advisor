#!/usr/bin/env bash
# 학습 PC의 활성 venv에서 CPU 고정을 같은 버전의 CUDA 휠로 교체한다.
set -euo pipefail

usage() {
    printf '%s\n' '사용법: scripts/install_cuda_torch.sh [--cuda cu130] [--dry-run]' \
        'TORCH_CUDA_VARIANT로 기본 CUDA 인덱스를 변경할 수 있다.' \
        '--dry-run은 설치와 GPU 검사 없이 실행할 명령만 출력한다.'
}

cuda_variant="${TORCH_CUDA_VARIANT:-cu130}"
dry_run=false
while (($#)); do
    case "$1" in
        --cuda)
            if (($# < 2)); then
                usage >&2
                exit 2
            fi
            cuda_variant="$2"
            shift 2
            ;;
        --dry-run) dry_run=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ ! "$cuda_variant" =~ ^cu[0-9]+$ ]]; then
    printf 'CUDA 인덱스는 cu130 같은 형식이어야 합니다: %s\n' "$cuda_variant" >&2
    exit 2
fi
if [[ -z "${VIRTUAL_ENV:-}" || ! -x "$VIRTUAL_ENV/bin/python" ]]; then
    printf '%s\n' '먼저 학습용 venv를 활성화하세요: source .venv/bin/activate' >&2
    exit 1
fi
python="$VIRTUAL_ENV/bin/python"
"$python" - <<'PY'
import os
from pathlib import Path
import sys

active = Path(os.environ["VIRTUAL_ENV"]).resolve()
if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != active:
    sys.exit("VIRTUAL_ENV가 실제 venv Python과 일치하지 않습니다.")
PY

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
requirements="$repo_root/requirements.txt"
run() {
    if "$dry_run"; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

run "$python" -m pip install -r "$requirements"
torch_version="$("$python" - "$requirements" <<'PY'
from pathlib import Path
import re
import sys

lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
pins = [
    line.strip()
    for line in lines
    if re.match(r"\s*torch(?=\s|[<>=!~@\[]|$)", line, re.I)
]
match = (
    re.fullmatch(r"torch\s*==\s*([0-9][A-Za-z0-9.!]*)\+cpu\s*(?:#.*)?", pins[0], re.I)
    if len(pins) == 1
    else None
)
if match is None:
    sys.exit("requirements.txt에 torch==<버전>+cpu 고정이 정확히 하나 있어야 합니다.")
print(match.group(1))
PY
)"
run "$python" -m pip install "torch==$torch_version+$cuda_variant" \
    --index-url "https://download.pytorch.org/whl/$cuda_variant"
if "$dry_run"; then
    printf '%s\n' 'dry-run: 설치 및 torch/CUDA 검사를 실행하지 않았습니다.'
    exit 0
fi

"$python" - "$torch_version+$cuda_variant" <<'PY'
import sys

import torch

available = torch.cuda.is_available()
print(f"torch version: {torch.__version__}")
print(f"torch.cuda.is_available(): {available}")
print(f"device name: {torch.cuda.get_device_name(0) if available else '없음'}")
if torch.__version__ != sys.argv[1]:
    sys.exit(f"요청한 CUDA 휠 버전과 다릅니다: expected {sys.argv[1]}")
if not available:
    sys.exit("CUDA를 사용할 수 없습니다. GPU 드라이버와 WSL GPU 연결을 확인하세요.")
PY
